import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

# =========================
# PATHS
# =========================
THIS_PATH = os.path.dirname(os.path.abspath(__file__))

abwaermepot_path = os.path.join(
    THIS_PATH,
    "raw_data",
    "Abwaermepotenzial_Adlershof_BfEE",
    "Abwaermepotenzial_Adlershof_BfEE.csv"
)

# TODO: We use the dummy central heat and cooling profile (centralheatprofile and
#  centralcoolprofile) here for now. Masking as scenario 2050, extr1, rcp85
#  For the model, as soon as the final demand profiles are generated
#  (Heat = RW + WW, Cool = Klima + Process) per 2035 and 2050 and Testreferenzjahr
#  the respective profile needs to be read in here. So the following code works only
#  for this one dummy time series scenario so far.

centralheatprofile_path = os.path.join(
    THIS_PATH,
    "raw_data",
    "Abwaerme_Profile",
    "demand_profiles",
    "2050_extr1_rcp85_central_HeatProfileNorm.csv"
)

centralcoolprofile_path = os.path.join(
    THIS_PATH,
    "raw_data",
    "Abwaerme_Profile",
    "demand_profiles",
    "2050_extr1_rcp85_central_CoolProfileNorm.csv"
)

wasteheatamounts_path = os.path.join(
    THIS_PATH,
    "raw_data",
    "Abwaerme_Profile",
    "waste_heat_amounts",
    "waste_heats.csv"
)

output_dir = os.path.join(THIS_PATH, "results", "Abwaerme_Profile")
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
abwaermepot = pd.read_csv(abwaermepot_path, sep=",")
centralheatprofile = pd.read_csv(centralheatprofile_path, sep=",")
centralcoolprofile = pd.read_csv(centralcoolprofile_path, sep=",")
waste_heats = pd.read_csv(wasteheatamounts_path, sep=",")

# =========================
# YEAR EXTRACTION
# =========================
def extract_year_from_filename(path):
    match = re.search(r"(20\d{2})", os.path.basename(path))
    if match:
        return match.group(1)
    else:
        raise ValueError("Kein Jahr im Dateinamen gefunden!")

year = extract_year_from_filename(centralheatprofile_path)
print(f"Verwendetes Jahr: {year}")

# =========================
# TIME INDEX
# =========================
centralheatprofile["datetime"] = pd.to_datetime(centralheatprofile["datetime"])
centralcoolprofile["datetime"] = pd.to_datetime(centralcoolprofile["datetime"])

df_time = pd.DataFrame(index=centralheatprofile["datetime"])

df_time["month"] = df_time.index.month
df_time["hour"] = df_time.index.hour
df_time["weekday"] = df_time.index.weekday

# =========================
# CENTRAL PROFILES
# =========================
central_heat = centralheatprofile["spaceHeatProfileNorm"].values
central_heat = central_heat / central_heat.sum()

central_cool = centralcoolprofile["CoolProfileNorm"].values
central_cool = central_cool / central_cool.sum()

# =========================
# TEMPERATURE CLASSIFICATION
# =========================
def classify_temp(temp_range):
    if temp_range in [">=110 °C", "90 - 110 °C"]:
        return "HT"
    elif temp_range in ["60 - 90 °C"]:
        return "MT"
    else:
        return "NT"

abwaermepot["Temp_Level"] = abwaermepot["Temperaturbereich"].apply(classify_temp)

# =========================
# MONTHS
# =========================
months = [
    "Leistungsprofil Januar (in kW)",
    "Leistungsprofil Februar (in kW)",
    "Leistungsprofil März (in kW)",
    "Leistungsprofil April (in kW)",
    "Leistungsprofil Mai (in kW)",
    "Leistungsprofil Juni (in kW)",
    "Leistungsprofil Juli (in kW)",
    "Leistungsprofil August (in kW)",
    "Leistungsprofil September (in kW)",
    "Leistungsprofil Oktober (in kW)",
    "Leistungsprofil November (in kW)",
    "Leistungsprofil Dezember (in kW)",
]

# =========================
# TIME WINDOWS
# =========================
time_windows = {
    "Kälteanlage-HUB-Kältezentrale": (0, 24),
    "NSHV": (0, 24),
    "zentrales Kühlsystem": (0, 24),
    "Luftkondensator": (0, 24),
    "Druckluft": (0, 24),
    "Abwasser": (0, 24),
    "Abwärme aus Gewerbekälteanlage": (0, 24),

    "iKWK Modul": (6, 17),
    "NEZ Modul": (7, 16),
    "Glasmodul": (8, 16),

    "KKM": (8, 16),
    "RLT": (8, 16),

    "KM": (7, 22),
    "Kälte BFS 360": (6, 18),
}

def get_time_window(name):
    for key in time_windows:
        if key in str(name):
            return time_windows[key]
    return (0, 24)

# =========================
# FUNCTIONS
# =========================
def get_monthly_weights(row):
    values = row[months].values.astype(float)
    if values.sum() == 0:
        return np.ones(12) / 12
    return values / values.sum()

def availability_mask(row):
    start, end = get_time_window(row["Name des Abwärmepotentials"])

    mask = (df_time["hour"] >= start) & (df_time["hour"] < end)
    mask = mask.astype(float)

    hours_per_day = row["Durchschnittliche tägl. Verfügbarkeit (in h)"]
    window_length = max(end - start, 1)

    mask *= min(hours_per_day / window_length, 1)

    if row["Verfügbarkeit am Wochenende"] == "Nein":
        mask[df_time["weekday"] >= 5] = 0

    return mask.values

def generate_profile(row):
    weights = get_monthly_weights(row)
    availability = availability_mask(row)

    total_energy = row["Wärmemenge pro Jahr (in kWh/a)"]
    profile = np.zeros(len(df_time))

    if row["Temp_Level"] == "NT":
        base_profile_global = central_cool
    else:
        base_profile_global = central_heat

    for m in range(1, 13):
        month_mask = (df_time["month"] == m).values

        base = base_profile_global * availability * month_mask

        if base.sum() == 0:
            base = base_profile_global * month_mask

        if base.sum() == 0:
            base = month_mask.astype(float)

        base = base / base.sum()
        monthly_energy = weights[m - 1] * total_energy

        profile += base * monthly_energy

    return profile

# =========================
# GENERATE PROFILES
# =========================
profiles = [generate_profile(row) for _, row in abwaermepot.iterrows()]
abwaermepot["profile"] = profiles

# =========================
# AGGREGATION
# =========================
def aggregate_profiles(df, level):
    subset = df[df["Temp_Level"] == level]
    if len(subset) == 0:
        return np.zeros(len(df_time))
    return np.sum(subset["profile"].tolist(), axis=0)

ht_profile = aggregate_profiles(abwaermepot, "HT")
mt_profile = aggregate_profiles(abwaermepot, "MT")
nt_profile = aggregate_profiles(abwaermepot, "NT")

# =========================
# TARGET VALUES FROM waste_heats
# =========================
def get_target(name):
    return waste_heats.loc[waste_heats["Abwärme"] == name, year].values[0]

ht_target = get_target("BTB-Abwärmerückgewinnung (Hochtemperatur)")
mt_target = get_target("Chemie + Industrie + BTB (Mitteltemperatur)")
nt_target = get_target("Rechenzentrum + BTB (Niedertemperatur) + Industrie (Niedertemperatur)")

# =========================
# SCALING
# =========================
def scale_profile(profile, target):
    if profile.sum() == 0:
        return profile
    return profile * (target / profile.sum())

ht_profile = scale_profile(ht_profile, ht_target)
mt_profile = scale_profile(mt_profile, mt_target)
nt_profile = scale_profile(nt_profile, nt_target)

# =========================
# VALIDATION
# Todo: This section is only for validation and debugging reasons.
#  Can be removed in model use.
# =========================
def validate(profile, expected, name):
    print(f"{name}: {profile.sum():.2f} / {expected:.2f} kWh")

validate(ht_profile, ht_target, "HT")
validate(mt_profile, mt_target, "MT")
validate(nt_profile, nt_target, "NT")

# =========================
# SAVE
# =========================
def save_profile(profile, name):
    df = pd.DataFrame({
        "datetime": df_time.index,
        "load_kWh": profile
    })
    df.to_csv(os.path.join(output_dir, f"{name}_profile.csv"), index=False)

save_profile(ht_profile, "HT")
save_profile(mt_profile, "MT")
save_profile(nt_profile, "NT")

# =========================
# PLOTS
# Todo: This section is only for validation and debugging reasons.
#  Can be removed in model use.
# =========================
plt.figure()
plt.plot(df_time.index, ht_profile)
plt.title("HT Profile")
plt.grid()
plt.show()

plt.figure()
plt.plot(df_time.index, mt_profile)
plt.title("MT Profile")
plt.grid()
plt.show()

plt.figure()
plt.plot(df_time.index, nt_profile)
plt.title("NT Profile")
plt.grid()
plt.show()

print("Fertig.")