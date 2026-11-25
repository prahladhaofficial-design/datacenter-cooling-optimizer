"""Lumped thermal model of a small air-cooled data-center hall (20 racks, hot-aisle / cold-aisle layout).

How air moves:
    The cooling unit (CRAH) blows cold air at the SUPPLY temperature T_sup into the cold aisle.
    Servers suck that air in (rack INLET), heat it by dT = P / (rho * cp * flow), and blow it into the hot aisle.
    Some hot exhaust leaks back over/around the racks into the cold aisle: RECIRCULATION.
    Racks at row ends get more recirculation, and if the CRAH supplies LESS air than the servers want
    (airflow ratio beta < 1) the shortfall is pulled from the hot aisle, so recirculation grows quickly.

Closed-form inlet temperature (per row, derived in the README):
    T_in,r = T_sup + rho_r * dT_row / (1 - mean(rho_row))
    rho_r  = recirculation fraction of rack r (0 = only cold air, 1 = only hot air)

Energy:
    chiller electric power = heat removed / COP(T_sup)
        COP curve of a chilled-water CRAC from HP Labs (Moore et al., USENIX 2005):
        COP(T) = 0.0068 T^2 + 0.0008 T + 0.458     (T = supply temperature in deg C)
        -> warmer supply air = much more efficient chiller.
    fan power ~ (airflow)^3   (fan affinity law)
    PUE (power usage effectiveness) = total facility power / IT power   (1.0 = perfect)

Constraint: every rack inlet must stay <= 27 C (upper end of the ASHRAE recommended inlet range, 18-27 C).
"""
from dataclasses import dataclass

import numpy as np

RHO_CP = 1.2 * 1005.0     # air density (kg/m3) * specific heat (J/kg/K)  -> J/(m3 K)
N_ROWS, RACKS_PER_ROW = 2, 10
N_RACKS = N_ROWS * RACKS_PER_ROW
INLET_LIMIT_C = 27.0


@dataclass
class Hall:
    rack_max_kw: np.ndarray = None       # nameplate power of each rack
    rho0: np.ndarray = None              # recirculation fraction at beta = 1
    server_dt_design: float = 12.0       # K temperature rise across servers at full power
    fan_kw_nominal: float = 12.0         # CRAH fan power at beta = 1
    recirc_sensitivity: float = 4.0      # how fast recirculation grows when air is short
    other_overhead: float = 0.06         # UPS losses, lighting... as fraction of IT power

    def __post_init__(self):
        rng = np.random.default_rng(7)
        if self.rack_max_kw is None:
            # mix of normal racks (~8 kW) and a few dense GPU racks (~14 kW)
            kw = np.full(N_RACKS, 8.0)
            kw[[3, 4, 14, 15]] = 14.0
            self.rack_max_kw = kw
        if self.rho0 is None:
            pos = np.tile(np.arange(RACKS_PER_ROW), N_ROWS)
            end = (pos == 0) | (pos == RACKS_PER_ROW - 1)
            near_end = (pos == 1) | (pos == RACKS_PER_ROW - 2)
            self.rho0 = 0.05 + 0.12 * end + 0.05 * near_end + rng.uniform(0, 0.03, N_RACKS)
        # server fans pull a fixed airflow sized for the design temperature rise
        self.rack_flow = self.rack_max_kw * 1000 / (RHO_CP * self.server_dt_design)  # m3/s
        self.total_flow = self.rack_flow.sum()

    # ------------------------------------------------------------------ physics
    def recirculation(self, beta):
        """beta = CRAH airflow / airflow the servers draw. Short air (beta<1) -> strong recirculation."""
        extra = np.exp(self.recirc_sensitivity * max(0.0, 1.0 - beta)) - 1.0
        relief = 1.0 / (1.0 + 0.8 * max(0.0, beta - 1.0))  # over-supply pushes hot air back a little
        return np.clip(self.rho0 * relief + 0.25 * extra * (self.rho0 / self.rho0.mean()), 0.0, 0.9)

    def inlet_temps(self, rack_kw, t_sup, beta):
        rho = self.recirculation(beta)
        t_in = np.empty(N_RACKS)
        for row in range(N_ROWS):
            s = slice(row * RACKS_PER_ROW, (row + 1) * RACKS_PER_ROW)
            dt_row = rack_kw[s].sum() * 1000 / (RHO_CP * self.rack_flow[s].sum())
            t_in[s] = t_sup + rho[s] * dt_row / (1.0 - rho[s].mean())
        return t_in

    def inlet_temps_grid(self, rack_kw, t_sup, beta):
        """Vectorised version: t_sup and beta are 1-D arrays of candidate set-points -> (n_candidates, N_RACKS)."""
        t_sup, beta = np.asarray(t_sup, float), np.asarray(beta, float)
        extra = np.exp(self.recirc_sensitivity * np.clip(1.0 - beta, 0, None)) - 1.0
        relief = 1.0 / (1.0 + 0.8 * np.clip(beta - 1.0, 0, None))
        rho = np.clip(self.rho0[None, :] * relief[:, None]
                      + 0.25 * extra[:, None] * (self.rho0 / self.rho0.mean())[None, :], 0.0, 0.9)
        t_in = np.empty((len(t_sup), N_RACKS))
        for row in range(N_ROWS):
            s = slice(row * RACKS_PER_ROW, (row + 1) * RACKS_PER_ROW)
            dt_row = rack_kw[s].sum() * 1000 / (RHO_CP * self.rack_flow[s].sum())
            t_in[:, s] = t_sup[:, None] + rho[:, s] * dt_row / (1.0 - rho[:, s].mean(axis=1, keepdims=True))
        return t_in

    def chiller_cop(self, t_sup):
        return 0.0068 * t_sup ** 2 + 0.0008 * t_sup + 0.458

    def power(self, rack_kw, t_sup, beta):
        it = rack_kw.sum()
        fan = self.fan_kw_nominal * beta ** 3
        heat = it + fan                        # fan energy also ends up as heat to remove
        chiller = heat / self.chiller_cop(t_sup)
        other = self.other_overhead * it
        total = it + fan + chiller + other
        return dict(it_kw=it, fan_kw=fan, chiller_kw=chiller, cooling_kw=fan + chiller,
                    total_kw=total, pue=total / it)


def workload(n_steps, seed=0, start_hour=0.0, step_h=0.25):
    """Synthetic but realistic IT utilisation: daily cycle (peak mid-afternoon), weekly dip, noise,
    and some racks busier than others. Rack power = 45 % idle + 55 % * utilisation (typical servers)."""
    rng = np.random.default_rng(seed)
    t = start_hour + step_h * np.arange(n_steps)
    hour, day = t % 24, (t // 24) % 7
    base = 0.45 + 0.30 * np.sin(2 * np.pi * (hour - 9) / 24) * (day < 5) + 0.12 * (day < 5)
    rack_bias = rng.uniform(-0.15, 0.15, N_RACKS)
    noise = rng.normal(0, 0.06, (n_steps, N_RACKS))
    util = np.clip(base[:, None] + rack_bias[None, :] + noise, 0.05, 1.0)
    return util


def rack_power(hall: Hall, util):
    return hall.rack_max_kw * (0.45 + 0.55 * util)
