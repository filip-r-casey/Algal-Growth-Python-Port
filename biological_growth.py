import numpy as np


def Temp_Efficiency_ORP(TR, T_opt, T_min, T_max):
    # Cardinal Model based on Rosso et al.1993
    if TR < T_min:
        Temp_eff = 0

    if TR >= T_min and TR <= T_max:
        g_of_t = (T_opt - T_min) * (TR - T_opt)
        f_of_t = (T_opt - T_max) * (T_opt + T_min - 2 * TR)
        Temp_eff = ((TR - T_max) * (TR - T_min) ** 2) / ((T_opt - T_min) * (g_of_t - f_of_t))

    if TR > T_max:
        Temp_eff = 0

    return Temp_eff


def Concentration_Efficiency_ORP(CX, ODC, depth):
    # This function determines the of impact of algal concentration on the
    # average light intensity hitting the culture

    gpl = CX / 1000  # gpl = grams per liter
    OD = gpl / ODC  # OD = optical density (750 nm)
    Conc_eff = (1 - np.exp(-depth * OD)) / (OD * depth)
    return Conc_eff


def Light_Efficiency_ORP(GHI, Conc_eff, I_sat):
    I_o = GHI * .45
    I_ave = I_o * Conc_eff
    Light_eff = (I_ave / I_sat) * np.exp(1 - (I_ave / I_sat))

    return Light_eff


def Night_Respiration_ORP(GHI, CX, night_resp, volume, Temp_eff):
    # based on Edmunson and Huessemann night respiration study. Study assumes 10-
    # hour dark period for the conversion from percentage to decay rate

    if GHI < 5:
        decay_rate = (np.log(1 - night_resp)) / 10  # yields
        decay_specific = (CX * (decay_rate / 3600))
        decay = (decay_specific * volume) * Temp_eff
    else:
        decay = 0

    return decay
