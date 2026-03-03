import pandas as pd
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# Path to directory with raw data
st_path = os.path.join(
    THIS_PATH,
    "raw_data",
    "Solarthermie_kWth",
    "Solathermieanlagen_verschnitten_Potenziale_ohne_Wohnen_am_Campus.csv"
)

# Read raw data
st = pd.read_csv(st_path, sep=",")

# Berechnung der thermischen Leistung in kWth
st["kWth"] = (
    st["kollektorf"] *
    st["sum_waerme"] /
    st["sum_modare"] /
    450
)

# Ausgabe
print(round(sum(st["kWth"]), 2))