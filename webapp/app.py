import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# -------------------------------------------------------------------
# CONFIG FICHIERS
# -------------------------------------------------------------------

# Fichier généré par export_recos_business.py
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
    / "recommandations_toners_latest.csv"
)

# Fichier où on garde les IDs "marqués comme envoyés"
PROCESSED_JSON = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "processed_recommendations_ui.json"
)

# Fichier KPAX nettoyé
KPAX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "kpax_consumables.parquet"
)

# -------------------------------------------------------------------
# COLONNES
# -------------------------------------------------------------------

COLUMN_SERIAL = "serial"
COLUMN_SERIAL_DISPLAY = "serial_display"
COLUMN_TONER = "toner"
COLUMN_DAYS = "jours_avant_rupture"
COLUMN_PRIORITY = "priorite"
COLUMN_CLIENT = "client"
COLUMN_CONTRACT = "contrat"
COLUMN_CITY = "ville"
COLUMN_COMMENT = "commentaire"
COLUMN_STOCKOUT = "date_rupture_estimee"
COLUMN_ID = "row_id"
COLUMN_TYPE_LIV = "type_livraison"

# KPAX dernières infos
COLUMN_LAST_UPDATE = "last_update"
COLUMN_LAST_PCT = "last_pct"

KPAX_STALE_DAYS = 20

# -------------------------------------------------------------------
# UTILITAIRES
# -------------------------------------------------------------------

def load_processed_ids():
    """Charge les IDs déjà marqués comme envoyé."""
    if PROCESSED_JSON.exists():
        try:
            with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed_ids", []))
        except Exception:
            return set()
    return set()


def save_processed_ids(processed_ids):
    """Sauvegarde les IDs envoyés."""
    PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"processed_ids": sorted(list(processed_ids))},
            f,
            ensure_ascii=False,
            indent=2,
        )

# -------------------------------------------------------------------
# CHARGEMENT KPAX POUR : dernière date + % par toner
# -------------------------------------------------------------------


def load_kpax_last_states():
    """
    Lis data/processed/kpax_consumables.parquet et retourne pour chaque
    (serial_display, couleur) :
      - la dernière date de remontée (last_update)
      - le % de toner à cette date (last_pct)
    """
    if not KPAX_PATH.exists():
        print("[KPAX] Fichier introuvable :", KPAX_PATH)
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    df = pd.read_parquet(KPAX_PATH)
    print("[KPAX] Colonnes trouvées :", list(df.columns))

    # Colonnes EXACTES présentes dans ton parquet
    serial_col = "$No serie$"
    date_update_col = "$Date update$"
    date_import_col = "$Date import$"

    color_cols = {
        "black": "$% noir$",
        "cyan": "$% cyan$",
        "magenta": "$% magenta$",
        "yellow": "$% jaune$",
    }

    # --- Fonction utilitaire pour parser les dates avec $08/02/23 08:11$ ---
    def parse_date(col_name):
        raw = (
            df[col_name]
            .astype(str)
            .str.strip()
            .str.strip("$")  # <- on enlève les $ au début / fin
        )
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)

    # 1️⃣ On tente d'abord avec $Date update$
    df["__date_update"] = parse_date(date_update_col)
    nb_valid_update = df["__date_update"].notna().sum()
    print(f"[KPAX] nb dates valides sur $Date update$ : {nb_valid_update}")

    if nb_valid_update > 0:
        df["__date_chosen"] = df["__date_update"]
    else:
        # 2️⃣ Sinon on prend $Date import$ en secours
        print("[KPAX] Fallback sur $Date import$")
        df["__date_import"] = parse_date(date_import_col)
        df["__date_chosen"] = df["__date_import"]

    long_parts = []
    for color_name, col_name in color_cols.items():
        if col_name not in df.columns:
            continue

        tmp = df[[serial_col, "__date_chosen", col_name]].copy()
        tmp = tmp.rename(
            columns={
                serial_col: COLUMN_SERIAL,
                "__date_chosen": COLUMN_LAST_UPDATE,
                col_name: COLUMN_LAST_PCT,
            }
        )

        # Nettoyage du serial → même format que dans le CSV métier
        tmp[COLUMN_SERIAL_DISPLAY] = (
            tmp[COLUMN_SERIAL]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.strip()
        )

        tmp["color"] = color_name

        # 🔧 NETTOYAGE DES % : "$98$" → "98" → 98.0
        pct_raw = (
            tmp[COLUMN_LAST_PCT]
            .astype(str)
            .str.strip()
            .str.strip("$")
        )
        tmp[COLUMN_LAST_PCT] = pd.to_numeric(pct_raw, errors="coerce")

        long_parts.append(tmp)

    if not long_parts:
        print("[KPAX] Aucune colonne couleur trouvée.")
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    long_df = pd.concat(long_parts, ignore_index=True)
    print("[KPAX] Après concat :", len(long_df))

    # On supprime les dates vides
    long_df = long_df.dropna(subset=[COLUMN_LAST_UPDATE])
    print("[KPAX] Après dropna date :", len(long_df))

    if long_df.empty:
        print("[KPAX] Aucune date valide ; retour DF vide.")
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    # On prend la dernière date par (serial_display, couleur)
    idx = long_df.groupby([COLUMN_SERIAL_DISPLAY, "color"])[COLUMN_LAST_UPDATE].idxmax()

    last_df = long_df.loc[
        idx,
        [COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT],
    ].copy()

    print("[KPAX] Derniers états calculés :", len(last_df))
    return last_df



KPAX_LAST_STATES = load_kpax_last_states()

# -------------------------------------------------------------------
# CHARGEMENT CSV BUSINESS + MERGE KPAX
# -------------------------------------------------------------------

def load_data():

    print("[DATA] Lecture :", DATA_PATH)

    df = pd.read_csv(DATA_PATH, sep=";")

    if COLUMN_ID not in df.columns:
        df[COLUMN_ID] = range(1, len(df) + 1)

    # Serial display pour l’affichage et les joins
    df[COLUMN_SERIAL_DISPLAY] = (
        df[COLUMN_SERIAL].astype(str).str.replace("$", "", regex=False).str.strip()
    )

    if COLUMN_DAYS in df.columns:
        df[COLUMN_DAYS] = pd.to_numeric(df[COLUMN_DAYS], errors="coerce")

    if COLUMN_PRIORITY in df.columns:
        df[COLUMN_PRIORITY] = (
            pd.to_numeric(df[COLUMN_PRIORITY], errors="coerce").astype("Int64")
        )

    # ------------------------------------------------------------
    # MERGE reco + KPAX (sur serial_display + couleur)
    # ------------------------------------------------------------
    if "couleur" in df.columns and not KPAX_LAST_STATES.empty:

        df["couleur_code"] = df["couleur"].astype(str).str.lower().str.strip()

        before = len(df)

        df = df.merge(
            KPAX_LAST_STATES,
            left_on=[COLUMN_SERIAL_DISPLAY, "couleur_code"],
            right_on=[COLUMN_SERIAL_DISPLAY, "color"],
            how="left",
        )

        after = len(df)
        print(f"[MERGE] avant={before} après={after}")

        df = df.drop(columns=["couleur_code", "color"], errors="ignore")

    else:
        print("[MERGE] Pas de couleur ou pas de KPAX_LAST_STATES → colonnes vides.")
        df[COLUMN_LAST_UPDATE] = pd.NaT
        df[COLUMN_LAST_PCT] = pd.NA

    # ------------------------------------------------------------
    # DÉTECTION KPAX MUET / RUPTURE POTENTIELLE
    # ------------------------------------------------------------
    today = pd.Timestamp.today().normalize()

    df[COLUMN_LAST_UPDATE] = pd.to_datetime(df[COLUMN_LAST_UPDATE], errors="coerce")
    df["days_since_last"] = (today - df[COLUMN_LAST_UPDATE]).dt.days

    df["rupture_kpax"] = (
        df["days_since_last"].isna()
        | (df["days_since_last"] > KPAX_STALE_DAYS)
    )

    return df

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():

    df = load_data()
    processed_ids = load_processed_ids()

    # Filtres
    serial_query = request.args.get("serial_query", "").strip()
    selected_priority = request.args.get("priority", "").strip()
    selected_type_liv = request.args.get("type_livraison", "").strip()

    pending_param = request.args.get("pending")
    show_only_pending = pending_param is None or pending_param == "1"

    if COLUMN_PRIORITY in df.columns:
        priorities = sorted(df[COLUMN_PRIORITY].dropna().unique().tolist())
    else:
        priorities = []

    if COLUMN_TYPE_LIV in df.columns:
        type_liv_options = (
            df[COLUMN_TYPE_LIV]
            .dropna()
            .astype(str)
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        type_liv_options = sorted(type_liv_options)
    else:
        type_liv_options = []

    filtered = df.copy()

    if serial_query:
        filtered = filtered[
            filtered[COLUMN_SERIAL_DISPLAY].str.contains(
                serial_query, case=False, na=False
            )
        ]

    if selected_priority:
        try:
            p_val = int(selected_priority)
            filtered = filtered[filtered[COLUMN_PRIORITY] == p_val]
        except ValueError:
            pass

    if selected_type_liv:
        filtered = filtered[filtered[COLUMN_TYPE_LIV] == selected_type_liv]

    if show_only_pending:
        filtered = filtered[~filtered[COLUMN_ID].isin(processed_ids)]

    sort_cols = []
    if COLUMN_PRIORITY in filtered.columns:
        sort_cols.append(COLUMN_PRIORITY)
    if COLUMN_DAYS in filtered.columns:
        sort_cols.append(COLUMN_DAYS)
    sort_cols.append(COLUMN_CLIENT)
    sort_cols.append(COLUMN_SERIAL_DISPLAY)

    # Les KPAX muets (rupture ?) doivent passer en dernier
    if "rupture_kpax" in filtered.columns:
        filtered = filtered.sort_values(
            by=["rupture_kpax"] + sort_cols,
            ascending=[True] + [True] * len(sort_cols),
        )
    else:
        filtered = filtered.sort_values(sort_cols)

    # --------- Regroupement par imprimante ----------
    grouped_printers = []
    if not filtered.empty:
        for serial_display, sub in filtered.groupby(COLUMN_SERIAL_DISPLAY):
            rows = sub.to_dict(orient="records")

            if COLUMN_DAYS in sub.columns:
                try:
                    min_days_left = float(sub[COLUMN_DAYS].min())
                except Exception:
                    min_days_left = None
            else:
                min_days_left = None

            if COLUMN_STOCKOUT in sub.columns:
                try:
                    min_stockout_date = str(sub[COLUMN_STOCKOUT].min())
                except Exception:
                    min_stockout_date = None
            else:
                min_stockout_date = None

            if COLUMN_PRIORITY in sub.columns:
                try:
                    min_priority = int(sub[COLUMN_PRIORITY].min())
                except Exception:
                    min_priority = None
            else:
                min_priority = None

            client_name = (
                sub[COLUMN_CLIENT].dropna().astype(str).iloc[0]
                if COLUMN_CLIENT in sub and not sub[COLUMN_CLIENT].isna().all()
                else None
            )
            contract_no = (
                sub[COLUMN_CONTRACT].dropna().astype(str).iloc[0]
                if COLUMN_CONTRACT in sub and not sub[COLUMN_CONTRACT].isna().all()
                else None
            )
            city = (
                sub[COLUMN_CITY].dropna().astype(str).iloc[0]
                if COLUMN_CITY in sub and not sub[COLUMN_CITY].isna().all()
                else None
            )

            grouped_printers.append(
                {
                    "serial_display": serial_display,
                    "client": client_name,
                    "contract": contract_no,
                    "city": city,
                    "rows": rows,
                    "min_days_left": min_days_left,
                    "min_stockout_date": min_stockout_date,
                    "min_priority": min_priority,
                }
            )

    # --------- Stats pour les graphiques ----------
    priority_counts = {}
    color_counts = {}

    if not filtered.empty and COLUMN_PRIORITY in filtered.columns:
        pr_series = filtered[COLUMN_PRIORITY].value_counts().sort_index()
        priority_counts = {str(int(k)): int(v) for k, v in pr_series.items()}

    if not filtered.empty and "couleur" in filtered.columns:
        col_series = filtered["couleur"].value_counts()
        color_counts = {str(k): int(v) for k, v in col_series.items()}

    # --------- Stats pour les chips ----------
    total_rows = int(len(filtered))
    pending_mask = ~filtered[COLUMN_ID].isin(processed_ids)
    pending_rows = int(pending_mask.sum())
    sent_rows = int((~pending_mask).sum())

    return render_template(
        "index.html",
        grouped_printers=grouped_printers,
        priorities=priorities,
        type_liv_options=type_liv_options,
        serial_query=serial_query,
        selected_priority=selected_priority,
        selected_type_liv=selected_type_liv,
        show_only_pending=show_only_pending,
        col_id=COLUMN_ID,
        col_toner=COLUMN_TONER,
        col_days=COLUMN_DAYS,
        col_priority=COLUMN_PRIORITY,
        col_client=COLUMN_CLIENT,
        col_contract=COLUMN_CONTRACT,
        col_city=COLUMN_CITY,
        col_comment=COLUMN_COMMENT,
        col_stockout=COLUMN_STOCKOUT,
        col_type_liv=COLUMN_TYPE_LIV,
        col_last_update=COLUMN_LAST_UPDATE,
        col_last_pct=COLUMN_LAST_PCT,
        processed_ids=processed_ids,
        priority_counts=priority_counts,
        color_counts=color_counts,
        total_rows=total_rows,
        pending_rows=pending_rows,
        sent_rows=sent_rows,
    )


@app.route("/mark_processed/<int:row_id>", methods=["POST"])
def mark_processed(row_id):
    processed = load_processed_ids()
    processed.add(row_id)
    save_processed_ids(processed)
    return redirect(url_for("index"))


@app.route("/mark_unprocessed/<int:row_id>", methods=["POST"])
def mark_unprocessed(row_id):
    processed = load_processed_ids()
    processed.discard(row_id)
    save_processed_ids(processed)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
