import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort

app = Flask(__name__)


@app.route("/health")
def health():
    return "ok", 200


# ============================================================
# CONFIG FICHIERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "outputs" / "recommandations_toners_latest.csv"
PROCESSED_JSON = BASE_DIR / "data" / "processed" / "processed_recommendations_ui.json"

# CSV léger (commité) : dernier état KPAX par (imprimante, couleur)
KPAX_LAST_CSV = BASE_DIR / "data" / "processed" / "kpax_last_states.csv"

# Parquet lourd (LOCAL seulement si tu l’as). Sur Render, ne pas le commit.
KPAX_PATH = BASE_DIR / "data" / "processed" / "kpax_consumables.parquet"

# Active/désactive l'historique (parquet KPAX) pour les graphes
KPAX_HISTORY_ENABLED = os.getenv("KPAX_HISTORY_ENABLED", "0") == "1"

FORECASTS_PATH = BASE_DIR / "data" / "processed" / "consumables_forecasts.parquet"

KPAX_HISTORY_CSV = BASE_DIR / "data" / "processed" / "kpax_history_light.csv"


# ============================================================
# COLONNES
# ============================================================

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

# ============================================================
# CACHES (IMPORTANT POUR RENDER)
# - On ne charge RIEN de lourd au moment de l'import du module
# ============================================================

_SLOPES_MAP = None
_KPAX_LAST_STATES = None
_KPAX_HISTORY_LONG = None


# ============================================================
# UTILITAIRES JSON (processed)
# ============================================================

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


# ============================================================
# SLOPES MAP (lazy-load)
# ============================================================

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


def get_slopes_map():
    global _SLOPES_MAP
    if _SLOPES_MAP is None:
        _SLOPES_MAP = load_slopes_map()
    return _SLOPES_MAP


# ============================================================
# KPAX - LAST STATES (CSV léger)
# ============================================================

def load_kpax_last_states():
    if not KPAX_LAST_CSV.exists():
        return pd.DataFrame(columns=["serial_display", "color", "last_update", "last_pct"])

    df = pd.read_csv(KPAX_LAST_CSV)
    df["serial_display"] = df["serial_display"].astype(str).str.strip()
    df["color"] = df["color"].astype(str).str.lower().str.strip()
    df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
    df["last_pct"] = pd.to_numeric(df["last_pct"], errors="coerce")
    return df


def get_kpax_last_states():
    global _KPAX_LAST_STATES
    if _KPAX_LAST_STATES is None:
        _KPAX_LAST_STATES = load_kpax_last_states()
    return _KPAX_LAST_STATES


# ============================================================
# KPAX - HISTORIQUE LONG (parquet lourd) - OPTIONNEL
# ============================================================

def load_kpax_history_long():
    """
    DF long: serial_display, color, date, pct
    - Si KPAX_HISTORY_ENABLED=False => renvoie vide (pas de parquet)
    - Sinon, lit KPAX_PATH (local) et met en cache
    """
    global _KPAX_HISTORY_LONG

    if not KPAX_HISTORY_ENABLED:
        return pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])

    if _KPAX_HISTORY_LONG is not None:
        return _KPAX_HISTORY_LONG

    if KPAX_HISTORY_CSV.exists():
        df = pd.read_csv(KPAX_HISTORY_CSV)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
        df["serial_display"] = df["serial_display"].astype(str).str.strip()
        df["color"] = df["color"].astype(str).str.lower().str.strip()
        return df.dropna(subset=["date","pct"])


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

    def parse_date(series: pd.Series):
        raw = series.astype(str).str.strip().str.strip("$")
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)

    d_up = parse_date(df[date_update_col]) if date_update_col in df.columns else pd.Series([pd.NaT] * len(df))
    d_im = parse_date(df[date_import_col]) if date_import_col in df.columns else pd.Series([pd.NaT] * len(df))
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
        }).dropna(subset=["date", "pct"])

        parts.append(tmp)

    if parts:
        long_df = pd.concat(parts, ignore_index=True)
        long_df = long_df.sort_values([COLUMN_SERIAL_DISPLAY, "color", "date"])
        _KPAX_HISTORY_LONG = long_df
    else:
        _KPAX_HISTORY_LONG = pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])

    print("[KPAX] Historique long chargé:", len(_KPAX_HISTORY_LONG))
    return _KPAX_HISTORY_LONG


def compute_slope_pct_per_day(dates: pd.Series, pcts: pd.Series):
    """Slope via régression linéaire pct ~ jours ; renvoie slope (%/jour)"""
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

    a, _b = np.polyfit(x, y, 1)
    return float(a)


# ============================================================
# CHARGEMENT CSV BUSINESS + MERGE KPAX (CSV léger)
# ============================================================

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

    kpax_last = get_kpax_last_states()

    if "couleur" in df.columns and not kpax_last.empty:
        df["couleur_code"] = df["couleur"].astype(str).str.lower().str.strip()

        before = len(df)
        df = df.merge(
            kpax_last,
            left_on=[COLUMN_SERIAL_DISPLAY, "couleur_code"],
            right_on=[COLUMN_SERIAL_DISPLAY, "color"],
            how="left",
        )
        after = len(df)
        print(f"[MERGE] avant={before} après={after}")

        df = df.drop(columns=["couleur_code", "color"], errors="ignore")
    else:
        print("[MERGE] Pas de couleur ou pas de KPAX_LAST_CSV → colonnes vides.")
        df[COLUMN_LAST_UPDATE] = pd.NaT
        df[COLUMN_LAST_PCT] = pd.NA

    # rupture flag + incohérence
    today = pd.Timestamp.today().normalize()
    df[COLUMN_LAST_UPDATE] = pd.to_datetime(df.get(COLUMN_LAST_UPDATE), errors="coerce")
    df["days_since_last"] = (today - df[COLUMN_LAST_UPDATE]).dt.days
    df["rupture_kpax"] = df["days_since_last"].isna() | (df["days_since_last"] > KPAX_STALE_DAYS)

    df["warning_incoherence"] = (
        (~df["rupture_kpax"])
        & df.get(COLUMN_LAST_PCT).notna()
        & (df.get(COLUMN_LAST_PCT) >= 20)
        & df.get(COLUMN_DAYS).notna()
        & (df.get(COLUMN_DAYS) <= 3)
    )

    if "couleur" in df.columns:
        df["couleur_norm"] = df["couleur"].astype(str).str.lower().str.strip()
    else:
        df["couleur_norm"] = ""

    return df


# ============================================================
# API
# ============================================================

@app.route("/api/consumption", methods=["GET"])
def api_consumption():
    serial = (request.args.get("serial") or "").strip()
    color = (request.args.get("color") or "").strip().lower()

    if not serial or not color:
        return jsonify({"error": "missing serial or color"}), 400

    slopes_map = get_slopes_map()

    # historique désactivé => renvoie slope forecasts + pas de points
    if not KPAX_HISTORY_ENABLED:
        slope_fallback = slopes_map.get((serial, color))
        return jsonify({
            "serial": serial,
            "color": color,
            "points": [],
            "slope_pct_per_day": slope_fallback,
            "slope_source": "forecasts_fallback" if slope_fallback is not None else None
        })

    hist = load_kpax_history_long()
    sub = hist[
        (hist[COLUMN_SERIAL_DISPLAY].astype(str) == serial)
        & (hist["color"].astype(str).str.lower() == color)
    ].copy()

    if sub.empty:
        slope_fallback = slopes_map.get((serial, color))
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
        slope = slopes_map.get((serial, color))
        slope_source = "forecasts_fallback" if slope is not None else None

    return jsonify({
        "serial": serial,
        "color": color,
        "points": points,
        "slope_pct_per_day": slope,
        "slope_source": slope_source
    })


@app.route("/api/slopes", methods=["GET"])
def api_slopes():
    serial = (request.args.get("serial") or "").strip()
    if not serial:
        return jsonify({"error": "missing serial"}), 400

    slopes_map = get_slopes_map()
    colors = ["black", "cyan", "magenta", "yellow"]
    slopes = {c: slopes_map.get((serial, c)) for c in colors}

    return jsonify({"serial": serial, "slopes": slopes})


@app.route("/api/printer_history", methods=["GET"])
def api_printer_history():
    serial = (request.args.get("serial") or "").strip()
    if not serial:
        return jsonify({"error": "missing serial"}), 400

    # historique désactivé => renvoie vide
    if not KPAX_HISTORY_ENABLED:
        return jsonify({"serial": serial, "series": {c: [] for c in ["black", "cyan", "magenta", "yellow"]}})

    hist = load_kpax_history_long()
    sub = hist[hist[COLUMN_SERIAL_DISPLAY].astype(str) == serial].copy().sort_values("date")

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


# ============================================================
# UI
# ============================================================

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
        filtered = filtered[filtered[COLUMN_SERIAL_DISPLAY].str.contains(serial_query, case=False, na=False)]

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

    rupture_mode = request.args.get("rupture_mode", "all")
    if rupture_mode == "rupture":
        filtered = filtered[filtered["rupture_kpax"] == True]
    elif rupture_mode == "ok":
        filtered = filtered[filtered["rupture_kpax"] == False]

    sort_cols = []
    if COLUMN_PRIORITY in filtered.columns:
        sort_cols.append(COLUMN_PRIORITY)
    if COLUMN_DAYS in filtered.columns:
        sort_cols.append(COLUMN_DAYS)
    sort_cols.append(COLUMN_CLIENT)
    sort_cols.append(COLUMN_SERIAL_DISPLAY)

    if "rupture_kpax" in filtered.columns:
        filtered = filtered.sort_values(by=["rupture_kpax"] + sort_cols, ascending=[True] + [True] * len(sort_cols))
    else:
        filtered = filtered.sort_values(sort_cols)

    grouped_printers = []
    if not filtered.empty:
        for serial_display, sub in filtered.groupby(COLUMN_SERIAL_DISPLAY):
            rows = sub.to_dict(orient="records")

            min_days_left = float(sub[COLUMN_DAYS].min()) if COLUMN_DAYS in sub.columns else None
            min_stockout_date = str(sub[COLUMN_STOCKOUT].min()) if COLUMN_STOCKOUT in sub.columns else None
            min_priority = int(sub[COLUMN_PRIORITY].min()) if COLUMN_PRIORITY in sub.columns and sub[COLUMN_PRIORITY].notna().any() else None

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

            grouped_printers.append({
                "serial_display": serial_display,
                "client": client_name,
                "contract": contract_no,
                "city": city,
                "rows": rows,
                "min_days_left": min_days_left,
                "min_stockout_date": min_stockout_date,
                "min_priority": min_priority,
                "colors_present": colors_present,
            })

    priority_counts = {}
    color_counts = {}

    if not filtered.empty and COLUMN_PRIORITY in filtered.columns:
        pr_series = filtered[COLUMN_PRIORITY].value_counts().sort_index()
        priority_counts = {str(int(k)): int(v) for k, v in pr_series.items()}

    if not filtered.empty and "couleur" in filtered.columns:
        col_series = filtered["couleur"].value_counts()
        color_counts = {str(k): int(v) for k, v in col_series.items()}

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


@app.route("/printer/<serial_display>", methods=["GET"])
def printer_detail(serial_display):
    serial_display = (serial_display or "").strip()

    df = load_data()
    sub = df[df[COLUMN_SERIAL_DISPLAY].astype(str) == serial_display].copy()
    if sub.empty:
        abort(404)

    client_name = sub[COLUMN_CLIENT].dropna().astype(str).iloc[0] if COLUMN_CLIENT in sub.columns and not sub[COLUMN_CLIENT].isna().all() else ""
    contract_no = sub[COLUMN_CONTRACT].dropna().astype(str).iloc[0] if COLUMN_CONTRACT in sub.columns and not sub[COLUMN_CONTRACT].isna().all() else ""
    city = sub[COLUMN_CITY].dropna().astype(str).iloc[0] if COLUMN_CITY in sub.columns and not sub[COLUMN_CITY].isna().all() else ""

    min_days_left = sub[COLUMN_DAYS].min() if COLUMN_DAYS in sub.columns else None
    min_stockout_date = sub[COLUMN_STOCKOUT].min() if COLUMN_STOCKOUT in sub.columns else None
    min_priority = sub[COLUMN_PRIORITY].min() if COLUMN_PRIORITY in sub.columns else None

    rows = sub.to_dict(orient="records")

    slopes_map = get_slopes_map()
    slopes = {c: slopes_map.get((serial_display, c)) for c in ["black", "cyan", "magenta", "yellow"]}

    # ---- KPAX STATUS (toujours depuis le CSV léger) ----
    kpax_last = get_kpax_last_states()
    ksub = kpax_last[kpax_last["serial_display"].astype(str) == serial_display].copy()

    kpax_status = []
    for c in ["black", "cyan", "magenta", "yellow"]:
        g = ksub[ksub["color"].astype(str).str.lower() == c]
        if g.empty:
            kpax_status.append({"color": c, "last_update": None, "last_pct": None})
        else:
            last = g.iloc[0]
            lu = last.get("last_update")
            kpax_status.append({
                "color": c,
                "last_update": lu.strftime("%Y-%m-%d") if pd.notna(lu) else None,
                "last_pct": float(last.get("last_pct")) if pd.notna(last.get("last_pct")) else None,
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
        kpax_history_enabled=KPAX_HISTORY_ENABLED,
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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
