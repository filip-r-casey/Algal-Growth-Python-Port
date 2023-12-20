import numpy as np

def Air_Properties(T):
    rho_air = 101.315 * 1000.0 / (287.058 * T)
    k_air = 0.02624 * np.power(T / 300, 0.8646)
    hfg_air = (-2E-05 * np.power(T, 3) + 0.0176 * np.power(T, 2) - 7.8474 * T + 3721.5)
    cp_air = 1002.5 + (275E-06) * np.power(T - 200, 2)
    dyn_visc_air = (1.458E-06) * np.power(T, 1.5) / (T + 110.4)
    pr_air = cp_air * dyn_visc_air / k_air
    spec_vol_air = 1 / rho_air
    return rho_air, k_air, hfg_air, cp_air, dyn_visc_air, pr_air, spec_vol_air
