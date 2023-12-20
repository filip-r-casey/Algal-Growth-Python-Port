import pandas as pd
from matplotlib import pyplot as plt
from data_collection import read_tmy_data
from location_data import get_tmy3_station
from parameters import Strain, Period
from timeutils import daterange
from biological_growth import Temp_Efficiency_ORP, Concentration_Efficiency_ORP, Light_Efficiency_ORP, \
    Night_Respiration_ORP
from thermal_model import Air_Properties
from rk4_fluxes import Direct_Solar_HT_ORP, Convection_HT_ORP, Longwave_Atmo_HT_ORP, Ground_Conduction_HT_ORP, \
    Evaporative_HT_ORP, Inflow_HT_ORP, Reradiation_HT_ORP
import datetime
import numpy as np


def simulation_runner(length):
    # Use a breakpoint in the code line below to debug your script.

    lat = 33.4942
    lon = -111.9261
    time_period = Period.annual_average.value
    n1 = datetime.datetime(2002, 1, 1)
    n2 = datetime.datetime(2002, 12, 31)

    # architecture Parameters
    growth_architecture = "Open Raceway Pond"
    num_ponds = 2
    width = 1.5
    length = length
    depth = .25
    panel_spacing = .5
    compass_orientation = 1.5708
    sparge_velocity = .002

    # strain-specific parameters
    microalgae_strain = Strain.chlorella_vulgaris.value
    T_opt = 300.5
    T_min = 278
    T_max = 315
    I_sat = 58
    ODC = .85
    night_resp = .0450

    # harvesting parameters
    IBC = 100
    harvest_conc = 300
    harvest_days = 7

    # CO2 consumption parameters (option)
    Mass_frac_CO2 = .50
    CO2_util = .40
    CO2_Source_Percent_CO2 = 1

    # nutrient consumption parameters
    Algae_N_Content = .0294
    Algae_P_Content = .0051
    Nut_surplus = .20

    # constants
    h = 3600
    rho_algae = 1000.0
    cp_algae = 4184.0
    Conversion = 4.56 * 10 ** -6
    area = np.pi * 0.25 * (width ** 2) + width * (length - width)
    volume = area * depth
    perimeter = (3.14 * (width / 2) ** 2) + (2 * (length - width))

    Ammonia_N_Content = 0.82
    DAP_N_Content = 0.18
    DAP_P_Content = 0.20

    Harvest_Mass = 0.0
    marker = 0
    marker_crash = 0
    op_hours = 0
    Freezes = 0

    TR = np.zeros((8760, 1))
    CX = np.zeros((8760, 1))
    Q_Atmospheric = np.zeros((8760, 1))
    M_Evapo = np.zeros((8760, 1))
    Q_Evapo = np.zeros((8760, 1))
    Q_Direct_Solar = np.zeros((8760, 1))
    Q_Ground_Conduction = np.zeros((8760, 1))
    Q_Reradiation = np.zeros((8760, 1))
    Q_Convection_track = np.zeros((8760, 1))
    Q_Makeup_Water = np.zeros((8760, 1))
    Water_consump = np.zeros((8760, 1))
    Temp_eff = np.zeros((8760, 1))
    Conc_eff = np.zeros((8760, 1))
    Light_eff = np.zeros((8760, 1))
    decay = np.zeros((8760, 1))
    Conc_at_Harvest = np.zeros((8760, 1))
    CO2_demand_hrly = np.zeros((8760, 1))
    Dried_Harvest = np.zeros((8760, 1))
    Harvest_Shortage = np.zeros((8760, 1))
    reliability = np.zeros((8760, 1))
    x = np.zeros((8760, 1))
    Crashes = np.zeros((8760, 1))
    Ammonia_demand_hrly = np.zeros((8760, 1))
    DAP_demand_hrly = np.zeros((8760, 1))
    Potash_demand_hrly = np.zeros((8760, 1))
    I_ave_culture = np.zeros((8760, 1))

    USAF = get_tmy3_station(lat, lon)
    tmy_df = read_tmy_data(USAF)
    # tmy_df = tmy_df.loc[(n1 <= tmy_df["datetime"]) & (tmy_df["datetime"] <= n2)].reset_index()
    for idx, row in tmy_df.iterrows():
        if idx == len(tmy_df) - 1:
            break
        if idx == 0:
            TR[idx] = row["T_amb"]
            CX[idx] = IBC
        elif CX[idx] == 0:
            CX[idx] = IBC
        else:
            TR[idx] = TR[idx]
            CX[idx] = CX[idx]

        # Biological Growth Model
        Temp_eff[idx] = Temp_Efficiency_ORP(TR[idx], T_opt, T_min, T_max)
        Conc_eff[idx] = Concentration_Efficiency_ORP(CX[idx], ODC, depth)
        Light_eff[idx] = Light_Efficiency_ORP(row["GHI"], Conc_eff[idx], I_sat)
        decay[idx] = Night_Respiration_ORP(row["GHI"], CX[idx], night_resp, volume, Temp_eff[idx])
        dCXdt = (12 / 8) * Light_eff[idx] * Temp_eff[idx] * row["GHI"] * .458 * .95 * Conversion * area + decay[idx]
        CX[idx + 1] = (dCXdt * h) / volume + CX[idx]

        # Thermal Model - 4th Order Runge-Kutta Scheme
        # Compute Air Properties at this time step using the film temperature
        rho_air, _, _, _, dyn_visc_air, _, _ = Air_Properties((TR[idx] + row["T_amb"]) / 2)
        # Compute thermal mass based on algal properties
        Therm_mass = volume * rho_algae * cp_algae

        # First RK4 Fluxes
        Q_solar = Direct_Solar_HT_ORP(row["GHI"])
        Q_Convection = Convection_HT_ORP(area, perimeter, TR[idx], row["T_amb"], row["WNDSPD"], dyn_visc_air, rho_air)
        Q_Longwave_Atmo = Longwave_Atmo_HT_ORP(row["T_amb"])
        Q_Ground = Ground_Conduction_HT_ORP(TR[idx])
        M_Evap, Q_Evap = Evaporative_HT_ORP(area, perimeter, TR[idx], row["T_amb"], row["RH"], row["WNDSPD"],
                                            dyn_visc_air, rho_air)
        Q_Inflow = Inflow_HT_ORP(M_Evap, cp_algae, TR[idx], row["T_amb"])
        Q_Rerad = Reradiation_HT_ORP(TR[idx])

        # RK4_1 time derivative of temperature - K1=dT1
        dT1 = (Q_Evap + Q_Convection + Q_Inflow + Q_Longwave_Atmo + Q_solar + Q_Ground + Q_Rerad) * area / Therm_mass
        T1 = TR[idx] + .5 * h * dT1

        # Air Properties - Thermal mass already defined
        rho_air, _, _, _, dyn_visc_air, _, _ = Air_Properties(
            (T1 + ((row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2)) / 2)

        Q_Solar = Direct_Solar_HT_ORP((row["GHI"] + tmy_df.iloc[idx + 1]["GHI"]) / 2)
        Q_Convection = Convection_HT_ORP(area, perimeter, T1, (row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2,
                                         (row["WNDSPD"] + tmy_df.iloc[idx + 1]["WNDSPD"]) / 2, dyn_visc_air, rho_air)
        Q_Longwave_Atmo = Longwave_Atmo_HT_ORP((row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2)
        Q_Ground = Ground_Conduction_HT_ORP(T1)
        M_Evap, Q_Evap = Evaporative_HT_ORP(area, perimeter, T1, (row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2,
                                            row["RH"] + tmy_df.iloc[idx + 1]["RH"] / 2,
                                            (row["WNDSPD"] + tmy_df.iloc[idx + 1]["WNDSPD"]) / 2,
                                            dyn_visc_air, rho_air)
        Q_Inflow = Inflow_HT_ORP(M_Evap, cp_algae, T1, (row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2)
        Q_Rerad = Reradiation_HT_ORP(T1)

        # RK4_2 time derivative - K2 = dT2
        dT2 = (Q_Evap + Q_Convection + Q_Inflow + Q_Longwave_Atmo + Q_Solar + Q_Ground + Q_Rerad) * area / Therm_mass
        T2 = TR[idx] + 0.5 * h * dT2

        # Second RK4 Fluxes
        Q_Convection = Convection_HT_ORP(area, perimeter, T1, ((row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2),
                                         (row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2, dyn_visc_air, rho_air)
        Q_Longwave_Atmo = Longwave_Atmo_HT_ORP((row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2)
        Q_Ground = Ground_Conduction_HT_ORP(T1)
        M_Evap, Q_Evap = Evaporative_HT_ORP(area, perimeter, T1, ((row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2),
                                            (row["RH"] + tmy_df.iloc[idx + 1]["RH"]) / 2,
                                            (row["WNDSPD"] + tmy_df.iloc[idx + 1]["WNDSPD"]) / 2,
                                            dyn_visc_air, rho_air)
        Q_Inflow = Inflow_HT_ORP(M_Evap, cp_algae, T1, (row["T_amb"] + tmy_df.iloc[idx + 1]["T_amb"]) / 2)
        Q_Rerad = Reradiation_HT_ORP(T1)

        # RK4_3 time derivative - K3 = dT3
        dT3 = (Q_Evap + Q_Convection + Q_Inflow + Q_Longwave_Atmo + Q_Ground + Q_Solar + Q_Rerad) * area / Therm_mass
        T3 = TR[idx] + h * dT3

        # Air Properties - Thermal mass already defined
        rho_air, _, _, _, dyn_visc_air, _, _ = Air_Properties((T3 + tmy_df.iloc[idx + 1]["T_amb"]) / 2)

        # Fourth RK4 Fluxes
        Q_Solar = Direct_Solar_HT_ORP(tmy_df.iloc[idx + 1]["GHI"])
        Q_Convection = Convection_HT_ORP(area, perimeter, T3, tmy_df.iloc[idx + 1]["T_amb"],
                                         tmy_df.iloc[idx + 1]["WNDSPD"], dyn_visc_air,
                                         rho_air)
        Q_Longwave_Atmo = Longwave_Atmo_HT_ORP(tmy_df.iloc[idx + 1]["T_amb"])
        Q_Ground = Ground_Conduction_HT_ORP(T3)
        M_Evap, Q_Evap = Evaporative_HT_ORP(area, perimeter, T3, tmy_df.iloc[idx + 1]["T_amb"],
                                            tmy_df.iloc[idx + 1]["RH"], tmy_df.iloc[idx + 1]["WNDSPD"],
                                            dyn_visc_air, rho_air)
        Q_Inflow = Inflow_HT_ORP(M_Evap, cp_algae, T3, tmy_df.iloc[idx + 1]["T_amb"])
        Q_Rerad = Reradiation_HT_ORP(T3)

        # RK4_1 time derivative
        dTdT = (Q_Evap + Q_Convection + Q_Inflow + Q_Longwave_Atmo + Q_Ground + Q_Solar + Q_Rerad) * area / Therm_mass
        # print(TR[idx], h, dT1, dT2, dT3, dTdT)
        TR[idx + 1] = TR[idx] + (1 / 6) * h * (dT1 + 2.0 * (dT2 + dT3) + dTdT)

        # Marker and Operational Hours Counting
        marker = marker + 1
        op_hours = op_hours + 1

        # %Harvesting Sequence
        if marker % (harvest_days * 24.0) == 0 or CX[idx + 1] >= harvest_conc or idx == (idx - 1):
            Harvest_Mass = Harvest_Mass + (CX[idx + 1] - IBC) * volume
            Conc_at_Harvest[idx] = CX[idx + 1]
            CX[idx + 1] = IBC
            marker = 0
        else:
            Harvest_Mass = Harvest_Mass + 0
            Conc_at_Harvest[idx] = 0
        # Track Thermal Fluxes with each Iteration
        rho_air, _, _, _, dyn_visc_air, _, _ = Air_Properties((TR[idx] + row["T_amb"]) / 2)
        Q_Atmospheric[idx] = Longwave_Atmo_HT_ORP(row["T_amb"])
        M_Evapo[idx], Q_Evapo[idx] = Evaporative_HT_ORP(area, perimeter, TR[idx], row["T_amb"], row["RH"],
                                                        row["WNDSPD"], dyn_visc_air, rho_air)
        Q_Makeup_Water[idx] = Inflow_HT_ORP(M_Evapo[idx], cp_algae, TR[idx], row["T_amb"])
        Q_Direct_Solar[idx] = Direct_Solar_HT_ORP(row["GHI"])
        Q_Ground_Conduction[idx] = Ground_Conduction_HT_ORP(TR[idx + 1])
        Q_Reradiation[idx] = Reradiation_HT_ORP(TR[idx + 1])
        Q_Convection_track[idx] = Convection_HT_ORP(area, perimeter, TR[idx + 1], row["T_amb"],
                                                    row["WNDSPD"],
                                                    dyn_visc_air, rho_air)
        I_ave_culture[idx] = row["GHI"] * 0.458 * Conc_eff[idx]

        # %Calculate waterloss at each time step
        Water_consump[idx] = M_Evapo[idx] * area * h  # [kg/m2*s]*[m2]*[3600 s] = [kg/hr]
        if Water_consump[idx] < 0:
            Water_consump[idx] = 0
        else:
            Water_consump[idx] = Water_consump[idx]

        # Calculate the CO2 demand kg/hr based on stoichiometric carbon balance
        if idx == 0:
            CO2_demand_hrly[idx] = 0
        else:
            CO2_demand_hrly[idx] = (((CX[idx + 1] - CX[idx]) * volume / 1000) * Mass_frac_CO2) / CO2_util
        if CO2_demand_hrly[idx] < 0:
            CO2_demand_hrly[idx] = 0
        else:
            CO2_demand_hrly[idx] = CO2_demand_hrly[idx]

        # Calculate the Nutrient consumption at each time step
        if idx == 0:
            Ammonia_demand_hrly[idx] = 0
            DAP_demand_hrly[idx] = 0
        else:
            DAP_demand_hrly[idx] = (((CX[idx + 1] - CX[
                idx]) * volume / 1000) * Algae_P_Content / DAP_P_Content);  # g/m3 * m3 * 1 kg/1000 g * %
            Ammonia_demand_hrly[idx] = (((CX[idx + 1] - CX[
                idx]) * volume / 1000) * Algae_N_Content / Ammonia_N_Content) - (DAP_demand_hrly[idx] * DAP_N_Content)

        if Ammonia_demand_hrly[idx] < 0:
            Ammonia_demand_hrly[idx] = 0
        else:
            Ammonia_demand_hrly[idx] = Ammonia_demand_hrly[idx] * (1 + Nut_surplus)

        if DAP_demand_hrly[idx] < 0:
            DAP_demand_hrly[idx] = 0
        else:
            DAP_demand_hrly[idx] = DAP_demand_hrly[idx] * (1 + Nut_surplus)

    # Conversion of Biomass to Fuel
    fuel_production = Harvest_Mass/ 4.18
    # https://www.researchgate.net/publication/309593471_Preliminary_assessment_of_Malaysian_micro-algae_strains_for_the_production_of_bio_jet_fuel

    aviation_fuel = fuel_production * 2

    # plane = "B737-800"
    # flight_length = 5  # hours
    # take_off_and_climb_fuel = 2300
    # active_burn = 2500
    # idle_burn = 300
    # fuel_usage = take_off_and_climb_fuel + flight_length * active_burn + 300

    number_of_flights = aviation_fuel/45359.237

    print(number_of_flights)

    # Results and Formatting
    # Summary of CO2 Consumption
    CO2_consump_total = np.sum(CO2_demand_hrly)

    plt.plot(CX)
    plt.xlabel("Timestamp (Hours)")
    plt.ylabel("Biomass (g * m^-3)")
    plt.title("Algal Biomass Cultivation in an Open Pond Structure over 1 year")
    plt.show()
    return number_of_flights


if __name__ == '__main__':
    flights_all = []
    for length in range(1, 100, 25):
        print(length)
        flights_all.append(simulation_runner(length))

    plt.scatter(np.arange(1, 100, 25), flights_all)
    plt.xlabel("ORP Length")
    plt.ylabel("Number of Flights")
    plt.title("Number of SAF Flights vs ORP Length")
    plt.show()
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
