import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KPAX_PATH = BASE_DIR / "data" / "processed" / "kpax_consumables.parquet"
OUT_CSV = BASE_DIR / "data" / "processed" / "kpax_history_light.csv"

KEEP_LAST_DAYS = 120  # mets 90 si tu veux encore plus léger

serial_col = "$No serie$"
date_update_col = "$Date update$"
date_import_col = "$Date import$"
color_cols = {
    "black": "$% noir$",
    "cyan": "$% cyan$",
    "magenta": "$% magenta$",
    "yellow": "$% jaune$",
}

cols_to_read = [serial_col, date_update_col, date_import_col] + list(color_cols.values())
df = pd.read_parquet(KPAX_PATH, columns=cols_to_read)

def parse_date(series):
    raw = series.astype(str).str.strip().str.strip("$")
    return pd.to_datetime(raw, errors="coerce", dayfirst=True)

d_up = parse_date(df[date_update_col])
d_im = parse_date(df[date_import_col])
date_chosen = d_up.where(d_up.notna(), d_im)

serial_display = df[serial_col].astype(str).str.replace("$", "", regex=False).str.strip()

parts = []
for color, col in color_cols.items():
    pct = pd.to_numeric(df[col].astype(str).str.strip().str.strip("$"), errors="coerce")
    tmp = pd.DataFrame({
        "serial_display": serial_display,
        "color": color,
        "date": date_chosen,
        "pct": pct
    }).dropna(subset=["date", "pct"])
    parts.append(tmp)

long_df = pd.concat(parts, ignore_index=True)
long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
long_df = long_df.dropna(subset=["date"])

cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=KEEP_LAST_DAYS)
long_df = long_df[long_df["date"] >= cutoff]

long_df = long_df.sort_values(["serial_display", "color", "date"])

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV, "rows:", len(long_df))
