"""Learn a fast ML surrogate of rack inlet temperatures, then use it to choose cooling set-points every 15 min.

Three ways to run the cooling for one week:
  1. Baseline   - what many halls do: fixed 18 C supply air, 20 % extra airflow "to be safe".
  2. Surrogate  - every 15 min, search all (supply temperature, airflow) pairs using the ML surrogate,
                  pick the lowest-power pair whose PREDICTED hottest inlet is <= 26 C (1 C safety margin
                  below the 27 C ASHRAE limit). The chosen set-point is then checked on the physics model.
  3. Oracle     - same search but with the physics model itself (best possible with this model).

Why a surrogate? In a real facility the "physics model" is a CFD simulation that takes minutes to hours,
far too slow to call thousands of times every 15 minutes. A neural-network surrogate trained on simulation
results answers in microseconds. Here the physics model is a fast lumped model, so the surrogate is a
demonstration of the workflow, and the oracle tells us how much the surrogate's errors cost.

Run:  python src/run_optimization.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dc_model import INLET_LIMIT_C, N_RACKS, Hall, rack_power, workload

RES = Path(__file__).resolve().parent.parent / "results"
RES.mkdir(exist_ok=True)
MARGIN = 1.0
T_GRID = np.arange(14.0, 27.01, 0.25)
B_GRID = np.arange(0.80, 1.401, 0.02)
TT, BB = [g.ravel() for g in np.meshgrid(T_GRID, B_GRID, indexing="ij")]


# ------------------------------------------------------------------ 1. surrogate
def make_dataset(hall, n, seed):
    rng = np.random.default_rng(seed)
    util = np.clip(rng.uniform(0.0, 1.0, (n, 1)) + rng.normal(0, 0.15, (n, N_RACKS)), 0.05, 1.0)
    P = rack_power(hall, util)
    t_sup = rng.uniform(13.0, 28.0, n)
    beta = rng.uniform(0.75, 1.45, n)
    X = np.column_stack([P, t_sup, beta])
    Y = np.vstack([hall.inlet_temps(P[i], t_sup[i], beta[i]) for i in range(n)])
    return X, Y


def train_surrogate(hall):
    X, Y = make_dataset(hall, 30000, seed=1)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, early_stopping=True,
                                       random_state=0))
    model.fit(Xtr, Ytr)
    pred = model.predict(Xte)
    stats = dict(train_samples=len(Xtr), test_samples=len(Xte),
                 inlet_mae_c=round(float(mean_absolute_error(Yte, pred)), 3),
                 inlet_r2=round(float(r2_score(Yte.ravel(), pred.ravel())), 4),
                 hottest_inlet_mae_c=round(float(np.mean(np.abs(Yte.max(1) - pred.max(1)))), 3))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Yte.max(1), pred.max(1), s=2, alpha=0.3)
    lim = [Yte.max(1).min(), Yte.max(1).max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("hottest inlet, physics model (°C)"); ax.set_ylabel("hottest inlet, NN surrogate (°C)")
    ax.set_title(f"Surrogate accuracy (test set), MAE {stats['hottest_inlet_mae_c']} °C")
    fig.tight_layout(); fig.savefig(RES / "surrogate_parity.png", dpi=150); plt.close(fig)
    return model, stats


# ------------------------------------------------------------------ 2. controllers
def grid_power(hall, p):
    it = p.sum()
    fan = hall.fan_kw_nominal * BB ** 3
    return fan + (it + fan) / hall.chiller_cop(TT)          # cooling power for every candidate


def choose(cooling_kw, hottest, limit):
    ok = hottest <= limit
    if not ok.any():
        return int(np.argmin(hottest))                      # nothing safe -> coldest option
    idx = np.flatnonzero(ok)
    return int(idx[np.argmin(cooling_kw[idx])])


def run_week(hall, surrogate):
    util = workload(7 * 96, seed=123)                       # unseen workload trace
    P = rack_power(hall, util)
    logs = {k: dict(t_sup=[], beta=[], pue=[], cooling_kw=[], it_kw=[], hottest=[])
            for k in ("Baseline (18 °C, 120 % air)", "NN-surrogate optimiser", "Oracle optimiser (physics model)")}
    for p in P:
        cool = grid_power(hall, p)
        Xg = np.column_stack([np.repeat(p[None, :], len(TT), 0), TT, BB])
        pred_hot = surrogate.predict(Xg).max(1)
        true_hot = hall.inlet_temps_grid(p, TT, BB).max(1)
        picks = {
            "Baseline (18 °C, 120 % air)": (18.0, 1.2),
            "NN-surrogate optimiser": (TT[choose(cool, pred_hot, INLET_LIMIT_C - MARGIN)],
                                       BB[choose(cool, pred_hot, INLET_LIMIT_C - MARGIN)]),
            "Oracle optimiser (physics model)": (TT[choose(cool, true_hot, INLET_LIMIT_C - MARGIN)],
                                                 BB[choose(cool, true_hot, INLET_LIMIT_C - MARGIN)]),
        }
        for name, (t, b) in picks.items():
            pw = hall.power(p, t, b)
            L = logs[name]
            L["t_sup"].append(t); L["beta"].append(b); L["pue"].append(pw["pue"])
            L["cooling_kw"].append(pw["cooling_kw"]); L["it_kw"].append(pw["it_kw"])
            L["hottest"].append(hall.inlet_temps(p, t, b).max())
    return {k: {m: np.array(v) for m, v in d.items()} for k, d in logs.items()}, P


def main():
    hall = Hall()
    print("training surrogate ...")
    surrogate, sstats = train_surrogate(hall)
    print(sstats)
    print("running one week of 15-minute control ...")
    logs, P = run_week(hall, surrogate)

    hours = 0.25
    summary = {}
    for name, L in logs.items():
        summary[name] = dict(
            mean_pue=round(float((L["cooling_kw"].sum() + L["it_kw"].sum() * (1 + hall.other_overhead)) / L["it_kw"].sum()), 3),
            cooling_energy_kwh=round(float(L["cooling_kw"].sum() * hours), 1),
            mean_supply_temp_c=round(float(L["t_sup"].mean()), 2),
            mean_airflow_ratio=round(float(L["beta"].mean()), 3),
            hottest_inlet_seen_c=round(float(L["hottest"].max()), 2),
            steps_over_27c=int((L["hottest"] > INLET_LIMIT_C).sum()),
        )
    base = summary["Baseline (18 °C, 120 % air)"]["cooling_energy_kwh"]
    for name in summary:
        summary[name]["cooling_energy_saving_pct"] = round(100 * (base - summary[name]["cooling_energy_kwh"]) / base, 1)
    out = dict(surrogate=sstats, week_simulation=summary, steps=len(P),
               it_energy_kwh=round(float(P.sum() * hours), 1))
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    plot_week(logs)
    plot_design_space(hall, P)


def plot_week(logs):
    colors = ["#999999", "#3b6ea5", "#2e8b57"]
    t = np.arange(len(next(iter(logs.values()))["pue"])) / 96
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for (name, L), c in zip(logs.items(), colors):
        axes[0].plot(t, L["pue"], color=c, lw=1, label=name)
        axes[1].plot(t, L["t_sup"], color=c, lw=1)
        axes[2].plot(t, L["hottest"], color=c, lw=1)
    axes[2].axhline(INLET_LIMIT_C, color="red", ls="--", lw=1, label="ASHRAE 27 °C limit")
    axes[0].set_ylabel("PUE"); axes[1].set_ylabel("supply air °C"); axes[2].set_ylabel("hottest rack inlet °C")
    axes[2].set_xlabel("day"); axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle("One week of cooling control, 20-rack hall (unseen workload)")
    fig.tight_layout(); fig.savefig(RES / "week_control.png", dpi=150); plt.close(fig)


def plot_design_space(hall, P):
    p = P[np.argmax(P.sum(1))]                              # busiest moment of the week
    cool = grid_power(hall, p).reshape(len(T_GRID), len(B_GRID))
    hot = hall.inlet_temps_grid(p, TT, BB).max(1).reshape(len(T_GRID), len(B_GRID))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(B_GRID, T_GRID, cool, levels=20, cmap="viridis")
    fig.colorbar(im, label="cooling power (kW)")
    cs = ax.contour(B_GRID, T_GRID, hot, levels=[INLET_LIMIT_C - MARGIN, INLET_LIMIT_C], colors=["white", "red"])
    ax.clabel(cs, fmt="%.0f °C")
    ax.set_xlabel("airflow ratio β (CRAH air / server air)"); ax.set_ylabel("supply air temperature (°C)")
    ax.set_title("Design space at peak IT load: cheapest safe point sits on the 26 °C line")
    fig.tight_layout(); fig.savefig(RES / "design_space.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
