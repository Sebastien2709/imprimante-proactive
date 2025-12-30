import pandas as pd
df = pd.read_csv("data/outputs/recommandations_toners_latest.csv", sep=";")

print(df.columns)
print(df.head())
print(df["couleur"].unique()[:20])


import pandas as pd
df = pd.read_parquet("data/processed/kpax_consumables.parquet")

print(df[["$% noir$", "$% cyan$", "$% magenta$", "$% jaune$"]].head())
print(df["$% noir$"].unique()[:20])
