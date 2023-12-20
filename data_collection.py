import pandas as pd
import datetime


def read_tmy_data(USAF):
    df = pd.read_csv("./data/tmyfiles/{usaf}TYA.csv".format(usaf=USAF), header=1)
    df = df.filter(items=["Date (MM/DD/YYYY)", "Time (HH:MM)", "Dry-bulb (C)", "GHI (W/m^2)", "RHum (%)", "Wspd (m/s)"])
    df = df.rename(
        columns={"Date (MM/DD/YYYY)": "Date", "Time (HH:MM)": "Time", "Dry-bulb (C)": "T_amb", "GHI (W/m^2)": "GHI",
                 "RHum (%)": "RH", "Wspd (m/s)": "WNDSPD"})
    df["datetime"] = pd.to_datetime(
        df["Date"] + "T" + df["Time"].apply(lambda x: "{:02d}".format(int(x.split(":")[0]) - 1)),
        format="%m/%d/%YT%H")
    df["T_amb"] = df["T_amb"] + 273.15
    df = df.drop(["Date", "Time"], axis=1)
    return df
