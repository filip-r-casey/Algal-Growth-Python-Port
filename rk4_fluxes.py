import numpy as np


def Direct_Solar_HT_ORP(GHI):
    # This function defines the heat flux from direct solar radiation
    f_a = .025
    Q_solar = (1 - f_a) * GHI * 0.82
    return Q_solar


def Convection_HT_ORP(area, perimeter, TR, T_amb, WNDSPD, dyn_visc_air, rho_air):
    # This function calculates the convection heat flux from the ORP
    kin_visc_a = dyn_visc_air / rho_air
    L_c = area / perimeter
    alpha_a = 2.2 * 10 ** -5
    lamda_a = 2.6 * 10 ** -2

    Re_L = L_c * WNDSPD / kin_visc_a
    Pr_a = kin_visc_a / alpha_a

    if Re_L < (3 * 10 ** 5):  # then the flow is laminar and use:
        Nu_L = 0.628 * (Re_L ** 0.5) * (Pr_a ** (1 / 3))
    elif Re_L > (5 * 10 ** 5):  # then the flow is turbulent and use:
        Nu_L = 0.035 * (Re_L ** 0.8) * (Pr_a ** (1 / 3))
    else:  # Average the values sherwood values for laminar and turbulent
        Nu_L = (0.628 * (Re_L ** 0.035 * (Re_L ** 0.8) * (Pr_a ** (1 / 3)))) / 2

    # Calculate the convection coefficient given the Nusselt number
    h_conv = Nu_L * lamda_a / L_c

    # Calculate the convective heat transfer
    Q_Convection = h_conv * (T_amb - TR)

    # Pretty much straight from Yadala and Cremaschi, 2016. Except for the added
    # correlation for laminar flow and averaging the two if in the transisition
    # period.
    return Q_Convection


def Longwave_Atmo_HT_ORP(T_amb):
    # This function defines the heat flux from longwave atmospheric radiation
    epsilon_water = 0.97  # emissivity of water under normal conditions
    epsilon_air = 0.85  # emissivity of air
    sigma = 5.67 * (10 ** -8)  # Stefan Boltzmann constant (W/m^2*k^4)
    Q_Longwave_Atmo = epsilon_water * epsilon_air * sigma * (T_amb ** 4);

    # This comes directly from Yadala and Cremaschi, 2016, however T_amb is
    # substituted for T_surr in the original equation. It is an approximation
    # to use T_amb
    return Q_Longwave_Atmo


def Ground_Conduction_HT_ORP(TR):
    # This function determines the heat flux between the ground and the ORP
    diff_concrete = 691.70 * (10 ** -9)
    k_concrete = 1.4
    l_ref = 4400 * diff_concrete ** 0.5
    Q_Ground = -k_concrete * (TR - 290.0) / l_ref

    # This heat flux equation comes from Bechet et al., "Universal temperature
    #  model for shallow algal ponds provides improved accuracy" and the
    #  equations are found in the SI document. It is approximating a depth in
    #  meters (3.65m) at which soil temp is unaffected by changes in ambient
    #  environment. Then it calculates a simple 1-D conduction through concrete
    #  using the thermal conductivty of concrete, and the the two temperatures
    #  (pond and soil). Soil temp at reference depth is assumed to equal 290K.

    return Q_Ground


def Evaporative_HT_ORP(area, perimeter, TR, T_amb, RH, WNDSPD, dyn_visc_air, rho_air):
    # This function calculates the evaporation rate of the pond as well as the
    # cooling effect due to that evaporation. Straight from Yadala and
    # Cremaschi, 2016.
    kin_visc_a = dyn_visc_air / rho_air
    L_c = area / perimeter
    D_w_a = 2.4 * 10 ** -5
    M_water = 0.018  # kg / mol
    R = 8.314  # Universal gas constant Pa * m3 / mol * K
    hfg_water = 2.45 * 10 ** 6

    # Calculate the Reynold's Number
    Re_L = L_c * WNDSPD / kin_visc_a
    Sch_L = kin_visc_a / D_w_a

    if Re_L < (3 * 10 ** 5):  # then the flow is laminar and use:
        Sh_L = 0.628 * (Re_L ** 0.5) * (Sch_L ** (1 / 3))
    elif Re_L > (5 * 10 ** 5):  # then the flow is turbulent and use:
        Sh_L = 0.035 * (Re_L ** 0.8) * (Sch_L ** (1 / 3))
    else:  # Average the values sherwood values for laminar and turbulent
        Sh_L = ((0.628 * (Re_L ** 0.5) * (Sch_L ** (1 / 3))) + (0.035 * (Re_L ** 0.8) * (Sch_L ** (1 / 3)))) / 2

    # Now with the sherwood number we can calculate mass transfer coefficient, K
    K = Sh_L * D_w_a / L_c

    # Calculate the saturated vapor pressures at T_amb and TR
    P_w = 3385.5 * np.exp(-8.0929 + 0.97608 * (TR + 42.607 - 273.15) ** 0.5)
    P_a = 3385.5 * np.exp(-8.0929 + 0.97608 * (T_amb + 42.607 - 273.15) ** 0.5)

    # Now we can calculate the evaporation rate in kg/m2*s
    M_Evap = K * ((P_w / TR) - (((RH / 100) * P_a) / T_amb)) * M_water / R  # Evaporation rate in kg / s * m2
    # print(K, P_w, TR, RH, P_a, T_amb, M_water, R)
    Q_Evap = -1.3 * M_Evap * hfg_water  # Evaporative heat transfer in W / m2,

    # This function calculates the evaporation rate of the pond as well as the
    # cooling effect due to that evaporation. Straight from Yadala and
    # Cremaschi, 2016.
    return M_Evap, Q_Evap


def Inflow_HT_ORP(M_Evap, cp_algae, TR, T_amb):
    # Calculates the Heat Flux due to the temperature difference between the
    # algae pond and the makeup water.

    Q_Inflow = cp_algae * M_Evap * (T_amb - TR)  # [J/kg*K] *[kg/m2*s] * [K] = [J/m2*s] = [w/m2]

    # Yadala and Cremaschi 2016
    return Q_Inflow


def Reradiation_HT_ORP(TR):
    # This function defines the heat flux from reradiation from the pond (Q_out)
    sigma = 5.67 * 10 ** -8
    epsilon_water = 0.97

    Q_Rerad = -1 * sigma * epsilon_water * (TR ** 4)

    # This function is the pond radiation. Yadala and Cremaschi, 2016. The
    # culture is approximated to have the emissivity of water.
    # Stephan-Boltzmann fourth power law estimation.

    return Q_Rerad
