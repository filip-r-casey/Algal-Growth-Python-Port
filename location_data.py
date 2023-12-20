from scipy import spatial
import pandas as pd


def get_tmy3_station(lat, lon):
    df = pd.read_csv("./data/tmy3stations.csv")
    tree = spatial.KDTree(df[["Latitude", "Longitude"]])
    return df.iloc[tree.query([lat, lon])[1]]["USAF"]
