import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort


app = Flask(__name__)

# -------------------------------------------------------------------
# CONFIG FICHIERS
# -------------------------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "outputs"
    / "recommandations_toners_latest.csv"
)

PROCESSED_JSON = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "processed_recommendations_ui.json"
)

KPAX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "kpax_consumables.parquet"
)

FORECASTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "consumables_forecasts.parquet"
)

# -------------------------------------------------------------------
# SLOPES MAP (fallback)
# -------------------------------------------------------------------

def load_slopes_map():
    """
    Charge les pentes depuis consumables_forecasts.parquet
    et renvoie {(serial_display, color): slope}
    """
    if not FORECASTS_PATH.exists():
        print("[SLOPES] Fichier introuvable:", FORECASTS_PATH)
        return {}

    df = pd.read_parquet(FORECASTS_PATH)
    cols = list(df.columns)

    # --- détecte colonne serial ---
    serial_col = None
    for cand in ["serial_display", "serial", "$no serie$"]:
        if cand in cols:
            serial_col = cand
            break
    if serial_col is None:
        for c in cols:
            cl = c.lower()
            if "serial" in cl or "serie" in cl:
                serial_col = c
                break

    # --- détecte colonne couleur ---
    color_col = None
    for cand in ["color", "couleur", "toner_color", "couleur_code"]:
        if cand in cols:
            color_col = cand
            break
    if color_col is None:
        for c in cols:
            cl = c.lower()
            if "color" in cl or "couleur" in cl:
                color_col = c
                break

    # --- détecte colonne pente ---
    slope_col = None
    for cand in ["slope", "slope_pct_per_day", "pct_per_day", "daily_slope"]:
        if cand in cols:
            slope_col = cand
            break
    if slope_col is None:
        for c in cols:
            if "slope" in c.lower():
                slope_col = c
                break

    if serial_col is None or color_col is None or slope_col is None:
        print("[SLOPES] Colonnes non détectées. Trouvées:", cols)
        return {}

    out = df[[serial_col, color_col, slope_col]].copy()

    out["serial_display"] = (
        out[serial_col].astype(str).str.replace("$", "", regex=False).str.strip()
    )
    out["color_norm"] = out[color_col].astype(str).str.lower().str.strip()
    out["slope_val"] = pd.to_numeric(out[slope_col], errors="coerce")
    out = out.dropna(subset=["serial_display", "color_norm", "slope_val"])

    slopes_map = {
        (r["serial_display"], r["color_norm"]): float(r["slope_val"])
        for _, r in out.iterrows()
    }

    print(f"[SLOPES] slopes chargées: {len(slopes_map)}")
    return slopes_map

SLOPES_MAP = load_slopes_map()

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
    if PROCESSED_JSON.exists():
        try:
            with open(PROCESSED_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed_ids", []))
        except Exception:
            return set()
    return set()

def save_processed_ids(processed_ids):
    PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"processed_ids": sorted(list(processed_ids))},
            f,
            ensure_ascii=False,
            indent=2,
        )

# -------------------------------------------------------------------
# KPAX - LAST STATES (comme chez toi)
# -------------------------------------------------------------------

def load_kpax_last_states():
    if not KPAX_PATH.exists():
        print("[KPAX] Fichier introuvable :", KPAX_PATH)
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    df = pd.read_parquet(KPAX_PATH)
    print("[KPAX] Colonnes trouvées :", list(df.columns))

    serial_col = "$No serie$"
    date_update_col = "$Date update$"
    date_import_col = "$Date import$"

    color_cols = {
        "black": "$% noir$",
        "cyan": "$% cyan$",
        "magenta": "$% magenta$",
        "yellow": "$% jaune$",
    }

    def parse_date(col_name):
        raw = df[col_name].astype(str).str.strip().str.strip("$")
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)

    df["__date_update"] = parse_date(date_update_col)
    nb_valid_update = df["__date_update"].notna().sum()
    print(f"[KPAX] nb dates valides sur $Date update$ : {nb_valid_update}")

    if nb_valid_update > 0:
        df["__date_chosen"] = df["__date_update"]
    else:
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

        tmp[COLUMN_SERIAL_DISPLAY] = (
            tmp[COLUMN_SERIAL].astype(str).str.replace("$", "", regex=False).str.strip()
        )
        tmp["color"] = color_name

        pct_raw = tmp[COLUMN_LAST_PCT].astype(str).str.strip().str.strip("$")
        tmp[COLUMN_LAST_PCT] = pd.to_numeric(pct_raw, errors="coerce")

        long_parts.append(tmp)

    if not long_parts:
        print("[KPAX] Aucune colonne couleur trouvée.")
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.dropna(subset=[COLUMN_LAST_UPDATE])

    if long_df.empty:
        return pd.DataFrame(
            columns=[COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]
        )

    idx = long_df.groupby([COLUMN_SERIAL_DISPLAY, "color"])[COLUMN_LAST_UPDATE].idxmax()
    last_df = long_df.loc[idx, [COLUMN_SERIAL_DISPLAY, "color", COLUMN_LAST_UPDATE, COLUMN_LAST_PCT]].copy()
    return last_df

KPAX_LAST_STATES = load_kpax_last_states()

# -------------------------------------------------------------------
# KPAX - HISTORIQUE LONG (pour le graphe)
# -------------------------------------------------------------------

_KPAX_HISTORY_LONG = None

def load_kpax_history_long():
    """
    Retourne un DF long avec colonnes:
      serial_display, color, date, pct
    Cache en mémoire pour éviter de relire le parquet à chaque click.
    """
    global _KPAX_HISTORY_LONG

    if _KPAX_HISTORY_LONG is not None:
        return _KPAX_HISTORY_LONG

    if not KPAX_PATH.exists():
        _KPAX_HISTORY_LONG = pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])
        return _KPAX_HISTORY_LONG

    df = pd.read_parquet(KPAX_PATH)

    serial_col = "$No serie$"
    date_update_col = "$Date update$"
    date_import_col = "$Date import$"

    color_cols = {
        "black": "$% noir$",
        "cyan": "$% cyan$",
        "magenta": "$% magenta$",
        "yellow": "$% jaune$",
    }

    def parse_date_series(col_name):
        raw = df[col_name].astype(str).str.strip().str.strip("$")
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)

    d_up = parse_date_series(date_update_col) if date_update_col in df.columns else pd.Series([pd.NaT]*len(df))
    d_im = parse_date_series(date_import_col) if date_import_col in df.columns else pd.Series([pd.NaT]*len(df))
    date_chosen = d_up.where(d_up.notna(), d_im)

    serial_display = df[serial_col].astype(str).str.replace("$", "", regex=False).str.strip()

    parts = []
    for color, col in color_cols.items():
        if col not in df.columns:
            continue
        pct_raw = df[col].astype(str).str.strip().str.strip("$")
        pct = pd.to_numeric(pct_raw, errors="coerce")

        tmp = pd.DataFrame({
            COLUMN_SERIAL_DISPLAY: serial_display,
            "color": color,
            "date": date_chosen,
            "pct": pct
        })
        tmp = tmp.dropna(subset=["date", "pct"])
        parts.append(tmp)

    if parts:
        long_df = pd.concat(parts, ignore_index=True)
        long_df = long_df.sort_values(["serial_display", "color", "date"])
        _KPAX_HISTORY_LONG = long_df
    else:
        _KPAX_HISTORY_LONG = pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])

    print("[KPAX] Historique long chargé:", len(_KPAX_HISTORY_LONG))
    return _KPAX_HISTORY_LONG

def compute_slope_pct_per_day(dates: pd.Series, pcts: pd.Series):
    """
    Slope via régression linéaire pct ~ jours (float) ; renvoie slope (%/jour)
    """
    if len(dates) < 2:
        return None

    d0 = dates.iloc[0]
    x = (dates - d0).dt.total_seconds() / 86400.0
    y = pcts.astype(float)

    mask = x.notna() & y.notna()
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()

    if x.size < 2:
        return None

    # y = a*x + b
    a, b = np.polyfit(x, y, 1)
    return float(a)

# -------------------------------------------------------------------
# CHARGEMENT CSV BUSINESS + MERGE KPAX (comme chez toi)
# -------------------------------------------------------------------

def load_data():
    print("[DATA] Lecture :", DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=";")

    if COLUMN_ID not in df.columns:
        df[COLUMN_ID] = range(1, len(df) + 1)

    df[COLUMN_SERIAL_DISPLAY] = (
        df[COLUMN_SERIAL].astype(str).str.replace("$", "", regex=False).str.strip()
    )

    if COLUMN_DAYS in df.columns:
        df[COLUMN_DAYS] = pd.to_numeric(df[COLUMN_DAYS], errors="coerce")

    if COLUMN_PRIORITY in df.columns:
        df[COLUMN_PRIORITY] = pd.to_numeric(df[COLUMN_PRIORITY], errors="coerce").astype("Int64")

    # Merge KPAX last states
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

        today = pd.Timestamp.today().normalize()
        df[COLUMN_LAST_UPDATE] = pd.to_datetime(df[COLUMN_LAST_UPDATE], errors="coerce")
        df["days_since_last"] = (today - df[COLUMN_LAST_UPDATE]).dt.days

        df["rupture_kpax"] = df["days_since_last"].isna() | (df["days_since_last"] > KPAX_STALE_DAYS)
        df = df.drop(columns=["couleur_code", "color"], errors="ignore")
    else:
        print("[MERGE] Pas de couleur ou pas de KPAX_LAST_STATES → colonnes vides.")
        df[COLUMN_LAST_UPDATE] = pd.NaT
        df[COLUMN_LAST_PCT] = pd.NA

    # rupture flag (sécurité)
    today = pd.Timestamp.today().normalize()
    df[COLUMN_LAST_UPDATE] = pd.to_datetime(df[COLUMN_LAST_UPDATE], errors="coerce")
    df["days_since_last"] = (today - df[COLUMN_LAST_UPDATE]).dt.days
    df["rupture_kpax"] = df["days_since_last"].isna() | (df["days_since_last"] > KPAX_STALE_DAYS)

    df["warning_incoherence"] = (
        (~df["rupture_kpax"])
        & df[COLUMN_LAST_PCT].notna()
        & (df[COLUMN_LAST_PCT] >= 20)
        & df[COLUMN_DAYS].notna()
        & (df[COLUMN_DAYS] <= 3)
    )

    # Normalisation couleur pour l’UI (utile pour les boutons)
    if "couleur" in df.columns:
        df["couleur_norm"] = df["couleur"].astype(str).str.lower().str.strip()
    else:
        df["couleur_norm"] = ""

    return df

# -------------------------------------------------------------------
# API : historique + pente
# -------------------------------------------------------------------

@app.route("/api/consumption", methods=["GET"])
def api_consumption():
    """
    Params:
      serial=XXXX
      color=black|cyan|magenta|yellow
    Renvoie:
      { serial, color, points:[{date,pct}], slope_pct_per_day, slope_source }
    """
    serial = (request.args.get("serial") or "").strip()
    color = (request.args.get("color") or "").strip().lower()

    if not serial or not color:
        return jsonify({"error": "missing serial or color"}), 400

    hist = load_kpax_history_long()
    sub = hist[
        (hist[COLUMN_SERIAL_DISPLAY].astype(str) == serial)
        & (hist["color"].astype(str).str.lower() == color)
    ].copy()

    if sub.empty:
        # fallback slope map only
        slope_fallback = SLOPES_MAP.get((serial, color))
        return jsonify({
            "serial": serial,
            "color": color,
            "points": [],
            "slope_pct_per_day": slope_fallback,
            "slope_source": "forecasts_fallback" if slope_fallback is not None else None
        })

    sub = sub.sort_values("date")
    points = [{"date": d.strftime("%Y-%m-%d"), "pct": float(p)} for d, p in zip(sub["date"], sub["pct"])]

    slope = compute_slope_pct_per_day(sub["date"].reset_index(drop=True), sub["pct"].reset_index(drop=True))
    slope_source = "kpax_regression"

    if slope is None:
        slope = SLOPES_MAP.get((serial, color))
        slope_source = "forecasts_fallback" if slope is not None else None

    return jsonify({
        "serial": serial,
        "color": color,
        "points": points,
        "slope_pct_per_day": slope,
        "slope_source": slope_source
    })

# -------------------------------------------------------------------
# ROUTES UI
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    df = load_data()
    processed_ids = load_processed_ids()

    serial_query = request.args.get("serial_query", "").strip()
    selected_priority = request.args.get("priority", "").strip()
    selected_type_liv = request.args.get("type_livraison", "").strip()

    pending_param = request.args.get("pending")
    show_only_pending = (pending_param == "1")


    priorities = sorted(df[COLUMN_PRIORITY].dropna().unique().tolist()) if COLUMN_PRIORITY in df.columns else []

    if COLUMN_TYPE_LIV in df.columns:
        type_liv_options = (
            df[COLUMN_TYPE_LIV].dropna().astype(str).replace("", pd.NA).dropna().unique().tolist()
        )
        type_liv_options = sorted(type_liv_options)
    else:
        type_liv_options = []

    filtered = df.copy()

    if serial_query:
        filtered = filtered[
            filtered[COLUMN_SERIAL_DISPLAY].str.contains(serial_query, case=False, na=False)
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

    if "rupture_kpax" in filtered.columns:
        filtered = filtered.sort_values(
            by=["rupture_kpax"] + sort_cols,
            ascending=[True] + [True] * len(sort_cols),
        )
    else:
        filtered = filtered.sort_values(sort_cols)

    rupture_mode = request.args.get("rupture_mode", "all")
    if rupture_mode == "rupture":
        filtered = filtered[filtered["rupture_kpax"] == True]
    elif rupture_mode == "ok":
        filtered = filtered[filtered["rupture_kpax"] == False]


    # --------- Regroupement par imprimante ----------
    grouped_printers = []
    if not filtered.empty:
        for serial_display, sub in filtered.groupby(COLUMN_SERIAL_DISPLAY):
            rows = sub.to_dict(orient="records")

            min_days_left = float(sub[COLUMN_DAYS].min()) if COLUMN_DAYS in sub.columns else None
            min_stockout_date = str(sub[COLUMN_STOCKOUT].min()) if COLUMN_STOCKOUT in sub.columns else None
            min_priority = int(sub[COLUMN_PRIORITY].min()) if COLUMN_PRIORITY in sub.columns else None

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

            # couleurs présentes pour cette imprimante (pour les boutons rapides / modal)
            colors_present = []
            if "couleur_norm" in sub.columns:
                colors_present = (
                    sub["couleur_norm"]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                colors_present = sorted(colors_present)

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
                    "colors_present": colors_present,
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

    # --------- Stats chips ----------
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
        rupture_mode=rupture_mode,
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
@app.route("/api/slopes", methods=["GET"])
def api_slopes():
    """
    Params:
      serial=XXXX (serial_display)
    Return:
      { serial, slopes: {black: -x, cyan: -y, magenta: -z, yellow: -w} }
    """
    serial = (request.args.get("serial") or "").strip()
    if not serial:
        return jsonify({"error": "missing serial"}), 400

    # Si on a déjà un SLOPES_MAP en mémoire (issu de consumables_forecasts.parquet)
    # on peut l'exploiter directement.
    colors = ["black", "cyan", "magenta", "yellow"]
    slopes = {}

    for c in colors:
        val = SLOPES_MAP.get((serial, c))
        slopes[c] = val  # peut être None

    # Si toutes les valeurs sont None, renvoyer quand même la structure
    return jsonify({
        "serial": serial,
        "slopes": slopes
    })

@app.route("/api/printer_history", methods=["GET"])
def api_printer_history():
    """
    Params:
      serial=XXXX (serial_display)
    Return:
      {
        serial: "....",
        series: {
          black: [{date:"YYYY-MM-DD", pct: 45.0}, ...],
          cyan: [...],
          magenta: [...],
          yellow: [...]
        }
      }
    """
    serial = (request.args.get("serial") or "").strip()
    if not serial:
        return jsonify({"error": "missing serial"}), 400

    hist = load_kpax_history_long()

    sub = hist[hist[COLUMN_SERIAL_DISPLAY].astype(str) == serial].copy()
    sub = sub.sort_values("date")

    series = {c: [] for c in ["black", "cyan", "magenta", "yellow"]}

    if not sub.empty:
        for color, g in sub.groupby("color"):
            color = str(color).lower().strip()
            if color not in series:
                continue
            series[color] = [
                {"date": d.strftime("%Y-%m-%d"), "pct": float(p)}
                for d, p in zip(g["date"], g["pct"])
                if pd.notna(d) and pd.notna(p)
            ]

    return jsonify({"serial": serial, "series": series})


@app.route("/printer/<serial_display>", methods=["GET"])

def printer_detail(serial_display):
    """
    Page détail imprimante: infos + tableau toners + graphe 4 couleurs
    """
    serial_display = (serial_display or "").strip()

    df = load_data()
    sub = df[df[COLUMN_SERIAL_DISPLAY].astype(str) == serial_display].copy()

    if sub.empty:
        abort(404)

    # Infos "header"
    client_name = sub[COLUMN_CLIENT].dropna().astype(str).iloc[0] if COLUMN_CLIENT in sub.columns and not sub[COLUMN_CLIENT].isna().all() else ""
    contract_no = sub[COLUMN_CONTRACT].dropna().astype(str).iloc[0] if COLUMN_CONTRACT in sub.columns and not sub[COLUMN_CONTRACT].isna().all() else ""
    city = sub[COLUMN_CITY].dropna().astype(str).iloc[0] if COLUMN_CITY in sub.columns and not sub[COLUMN_CITY].isna().all() else ""

    # Quelques agrégats utiles
    min_days_left = sub[COLUMN_DAYS].min() if COLUMN_DAYS in sub.columns else None
    min_stockout_date = sub[COLUMN_STOCKOUT].min() if COLUMN_STOCKOUT in sub.columns else None
    min_priority = sub[COLUMN_PRIORITY].min() if COLUMN_PRIORITY in sub.columns else None

    # Toners / lignes
    rows = sub.to_dict(orient="records")

    # Pentes (si tu veux aussi les afficher sur la page détail)
    slopes = {c: SLOPES_MAP.get((serial_display, c)) for c in ["black", "cyan", "magenta", "yellow"]}

    # --- KPAX status (4 couleurs) depuis l'historique (robuste) ---
    hist = load_kpax_history_long()
    hsub = hist[hist[COLUMN_SERIAL_DISPLAY].astype(str) == serial_display].copy()
    hsub = hsub.dropna(subset=["date", "pct"]).sort_values("date")

    kpax_status = []
    for c in ["black", "cyan", "magenta", "yellow"]:
        g = hsub[hsub["color"].astype(str).str.lower() == c]
        if g.empty:
            kpax_status.append({"color": c, "last_update": None, "last_pct": None})
        else:
            last = g.iloc[-1]
            kpax_status.append({
                "color": c,
                "last_update": last["date"].strftime("%Y-%m-%d"),
                "last_pct": float(last["pct"]),
            })


    return render_template(
        "printer_detail.html",
        serial_display=serial_display,
        client=client_name,
        contract=contract_no,
        city=city,
        min_days_left=min_days_left,
        min_stockout_date=min_stockout_date,
        min_priority=min_priority,
        rows=rows,
        slopes=slopes,
        kpax_status=kpax_status,
        col_id=COLUMN_ID,
        col_toner=COLUMN_TONER,
        col_days=COLUMN_DAYS,
        col_priority=COLUMN_PRIORITY,
        col_comment=COLUMN_COMMENT,
        col_stockout=COLUMN_STOCKOUT,
        col_type_liv=COLUMN_TYPE_LIV,
        col_last_update=COLUMN_LAST_UPDATE,
        col_last_pct=COLUMN_LAST_PCT,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
