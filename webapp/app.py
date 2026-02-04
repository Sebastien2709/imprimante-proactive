import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort

app = Flask(__name__)


def load_kpax_history_for_serial(serial: str) -> pd.DataFrame:
    """
    Lit kpax_history_light.csv en CHUNKS et ne garde en mémoire
    que les lignes du serial demandé.
    """
    serial = (serial or "").strip()
    if not serial or not KPAX_HISTORY_CSV.exists():
        return pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])

    usecols = ["serial_display", "color", "date", "pct"]
    chunks = []
    for chunk in pd.read_csv(KPAX_HISTORY_CSV, usecols=usecols, chunksize=250_000):
        # normalisation light (important pour match)
        chunk["serial_display"] = chunk["serial_display"].astype(str).str.strip()
        sub = chunk[chunk["serial_display"] == serial]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])

    df = pd.concat(chunks, ignore_index=True)
    df["color"] = df["color"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df = df.dropna(subset=["date", "pct"])
    return df


@app.route("/api/debug_kpax", methods=["GET"])
def api_debug_kpax():
    info = {
        "BASE_DIR": str(BASE_DIR),
        "KPAX_HISTORY_CSV": str(KPAX_HISTORY_CSV),
        "KPAX_HISTORY_CSV_exists": KPAX_HISTORY_CSV.exists(),
        "KPAX_LAST_CSV": str(KPAX_LAST_CSV),
        "KPAX_LAST_CSV_exists": KPAX_LAST_CSV.exists(),
    }
    if KPAX_HISTORY_CSV.exists():
        try:
            df = pd.read_csv(KPAX_HISTORY_CSV)
            info["kpax_history_rows"] = int(len(df))
            info["kpax_history_cols"] = list(df.columns)
            # montre 2 exemples
            info["kpax_history_head"] = df.head(2).to_dict(orient="records")
        except Exception as e:
            info["kpax_history_read_error"] = str(e)
    return jsonify(info)


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

# CSV léger (commité) : historique KPAX (format long) pour le graphe (safe Render)
KPAX_HISTORY_CSV = BASE_DIR / "data" / "processed" / "kpax_history_light.csv"

# Parquet lourd (LOCAL seulement si tu l'as). Sur Render, ne pas le commit.
KPAX_PATH = BASE_DIR / "data" / "processed" / "kpax_consumables.parquet"

# Active/désactive l'historique via parquet (LOCAL)
# Sur Render tu laisses à 0
KPAX_HISTORY_ENABLED = os.getenv("KPAX_HISTORY_ENABLED", "0") == "1"

FORECASTS_PATH = BASE_DIR / "data" / "processed" / "consumables_forecasts.parquet"


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

# Colonnes statut contrat
COLUMN_STATUT_CONTRAT = "statut_contrat"
COLUMN_DATE_FIN_CONTRAT = "date_fin_contrat"

# KPAX dernières infos
COLUMN_LAST_UPDATE = "last_update"
COLUMN_LAST_PCT = "last_pct"

KPAX_STALE_DAYS = 20

# ============================================================
# CACHES (IMPORTANT POUR RENDER)
# ============================================================

_SLOPES_MAP = None
_KPAX_LAST_STATES = None
_KPAX_HISTORY_LONG = None
_DATA_CACHE = None          # (mtime, DataFrame)
_CONTRACT_CACHE = None      # (mtime, DataFrame)


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
    out["serial_display"] = out[serial_col].astype(str).str.replace("$", "", regex=False).str.strip()
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
# STATUTS DE CONTRATS (ItemLedgEntries)
# ============================================================

def _load_contract_statuses_from_disk():
    """Lecture brute du parquet — appelée une seule fois puis cachée."""
    contracts_path = BASE_DIR / "data" / "processed" / "contract_status.parquet"

    if not contracts_path.exists():
        print(f"[CONTRATS] Fichier introuvable : {contracts_path}")
        print("[CONTRATS] Exécutez d'abord : python src/data/load_contract_status.py")
        return pd.DataFrame(columns=["serial_display", COLUMN_STATUT_CONTRAT, COLUMN_DATE_FIN_CONTRAT])

    try:
        print(f"[CONTRATS] Lecture disque : {contracts_path}")
        df = pd.read_parquet(contracts_path)

        if "statut_contrat" in df.columns and COLUMN_STATUT_CONTRAT not in df.columns:
            df = df.rename(columns={"statut_contrat": COLUMN_STATUT_CONTRAT})
        if "date_fin_contrat" in df.columns and COLUMN_DATE_FIN_CONTRAT not in df.columns:
            df = df.rename(columns={"date_fin_contrat": COLUMN_DATE_FIN_CONTRAT})

        print(f"[CONTRATS] {len(df)} contrats chargés")
        return df
    except Exception as e:
        print(f"[CONTRATS] Erreur lors du chargement : {e}")
        return pd.DataFrame(columns=["serial_display", COLUMN_STATUT_CONTRAT, COLUMN_DATE_FIN_CONTRAT])


def load_contract_statuses():
    """Cache par mtime — ne relit le parquet que si le fichier a changé."""
    global _CONTRACT_CACHE
    contracts_path = BASE_DIR / "data" / "processed" / "contract_status.parquet"

    if contracts_path.exists():
        mtime = contracts_path.stat().st_mtime
    else:
        mtime = 0

    if _CONTRACT_CACHE is not None:
        cached_mtime, cached_df = _CONTRACT_CACHE
        if cached_mtime == mtime:
            return cached_df.copy()

    df = _load_contract_statuses_from_disk()
    _CONTRACT_CACHE = (mtime, df)
    return df.copy()


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
# KPAX - HISTORIQUE LONG
# - Priorité au CSV léger (safe Render)
# - Sinon parquet lourd (local) si KPAX_HISTORY_ENABLED=1
# ============================================================

def load_kpax_history_long():
    global _KPAX_HISTORY_LONG

    if _KPAX_HISTORY_LONG is not None:
        return _KPAX_HISTORY_LONG

    # 1) CSV léger (recommandé, fonctionne sur Render)
    if KPAX_HISTORY_CSV.exists():
        df = pd.read_csv(KPAX_HISTORY_CSV)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
        df["serial_display"] = df["serial_display"].astype(str).str.strip()
        df["color"] = df["color"].astype(str).str.lower().str.strip()
        df = df.dropna(subset=["date", "pct"])
        _KPAX_HISTORY_LONG = df
        print("[KPAX] Historique chargé depuis CSV:", KPAX_HISTORY_CSV, "rows:", len(df))
        return _KPAX_HISTORY_LONG

    # 2) Sinon, historique désactivé -> pas de points
    if not KPAX_HISTORY_ENABLED:
        _KPAX_HISTORY_LONG = pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])
        return _KPAX_HISTORY_LONG

    # 3) Parquet lourd (local)
    if not KPAX_PATH.exists():
        _KPAX_HISTORY_LONG = pd.DataFrame(columns=[COLUMN_SERIAL_DISPLAY, "color", "date", "pct"])
        return _KPAX_HISTORY_LONG

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

def _load_data_from_disk():
    print("[DATA] Lecture :", DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=";")

    if COLUMN_ID not in df.columns:
        df[COLUMN_ID] = range(1, len(df) + 1)

    df[COLUMN_SERIAL_DISPLAY] = df[COLUMN_SERIAL].astype(str).str.replace("$", "", regex=False).str.strip()

    if COLUMN_DAYS in df.columns:
        df[COLUMN_DAYS] = pd.to_numeric(df[COLUMN_DAYS], errors="coerce").round(0).astype("Int64")

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

    # ============================================================
    # MERGE STATUTS DE CONTRATS
    # ============================================================
    contracts = load_contract_statuses()
    
    if not contracts.empty:
        # Renommer serial_norm en serial_display pour le merge
        if "serial_norm" in contracts.columns and "serial_display" not in contracts.columns:
            contracts = contracts.rename(columns={"serial_norm": "serial_display"})
        
        before = len(df)
        df = df.merge(
            contracts,
            on="serial_display",
            how="left"
        )
        after = len(df)
        print(f"[MERGE CONTRATS] avant={before} après={after}")
    else:
        print("[MERGE CONTRATS] Pas de fichier contrats → colonnes vides.")
        df[COLUMN_STATUT_CONTRAT] = pd.NA
        df[COLUMN_DATE_FIN_CONTRAT] = pd.NaT

    # ============================================================
    # FILTRAGE SELON STATUT CONTRAT
    # ============================================================
    # Cas 1: Annulé sans date → SUPPRIMER complètement
    mask_annule_sans_date = (
        df[COLUMN_STATUT_CONTRAT].astype(str).str.lower().str.contains("annul", na=False) &
        df[COLUMN_DATE_FIN_CONTRAT].isna()
    )
    nb_supprime = mask_annule_sans_date.sum()
    if nb_supprime > 0:
        print(f"[CONTRATS] {nb_supprime} lignes supprimées (contrat annulé sans date)")
        df = df[~mask_annule_sans_date].copy()
    
    # DEBUG: Afficher les statuts trouvés
    if COLUMN_STATUT_CONTRAT in df.columns:
        statut_counts = df[COLUMN_STATUT_CONTRAT].value_counts(dropna=False).to_dict()
        print(f"[CONTRATS DEBUG] Statuts présents : {statut_counts}")

    # Cas 2: Annulé avec date → Flag "alerte_non_prioritaire"
    df["alerte_non_prioritaire"] = (
        df[COLUMN_STATUT_CONTRAT].astype(str).str.lower().str.contains("annul", na=False) &
        df[COLUMN_DATE_FIN_CONTRAT].notna()
    )
    
    # Cas 3: Signé avec date proche (< 30 jours) → Flag "alerte_contrat_fin_proche"
    df["alerte_contrat_fin_proche"] = False
    if COLUMN_DATE_FIN_CONTRAT in df.columns:
        mask_signe = df[COLUMN_STATUT_CONTRAT].astype(str).str.lower().str.contains("sign", na=False)
        mask_date_proche = df[COLUMN_DATE_FIN_CONTRAT].notna() & ((df[COLUMN_DATE_FIN_CONTRAT] - today).dt.days <= 30)
        df.loc[mask_signe & mask_date_proche, "alerte_contrat_fin_proche"] = True

    # Ajouter commentaires d'alerte
    df.loc[df["alerte_non_prioritaire"], COLUMN_COMMENT] = (
        "⚠️ Contrat annulé - Fin prévue le " + 
        df.loc[df["alerte_non_prioritaire"], COLUMN_DATE_FIN_CONTRAT].dt.strftime("%d/%m/%Y")
    )
    
    df.loc[df["alerte_contrat_fin_proche"], COLUMN_COMMENT] = (
        "⏰ Contrat signé - Fin prévue le " + 
        df.loc[df["alerte_contrat_fin_proche"], COLUMN_DATE_FIN_CONTRAT].dt.strftime("%d/%m/%Y")
    )

    nb_alertes_non_prio = df["alerte_non_prioritaire"].sum()
    nb_alertes_fin_proche = df["alerte_contrat_fin_proche"].sum()
    print(f"[CONTRATS] {nb_alertes_non_prio} alertes non prioritaires, {nb_alertes_fin_proche} contrats fin proche")

    # ============================================================
    # AJOUT "TONER BAS MAIS NON PRIORITAIRE" (KPAX < 3%)
    # - Ne touche PAS au ML
    # - Ajoute une ligne uniquement si (serial, couleur) pas deja recommande
    # - Force priorite = 4
    # - UNIQUEMENT si contrat de maintenance existe
    # ============================================================
    
    if not kpax_last.empty and "couleur_norm" in df.columns:
        # Couples deja presents dans les recos (ML/business)
        existing_pairs = set(
            zip(
                df[COLUMN_SERIAL_DISPLAY].astype(str).str.strip(),
                df["couleur_norm"].astype(str).str.lower().str.strip(),
            )
        )

        # Pré-compiler un dict serial → (client, contrat, ville, type_liv)
        # pour éviter de filtrer le DF à chaque itération
        _serial_info = {}
        for serial_val, grp in df.groupby(COLUMN_SERIAL_DISPLAY):
            _serial_info[str(serial_val).strip()] = {
                "client": str(grp[COLUMN_CLIENT].dropna().iloc[0]) if COLUMN_CLIENT in grp.columns and not grp[COLUMN_CLIENT].isna().all() else "",
                "contract": str(grp[COLUMN_CONTRACT].dropna().iloc[0]) if COLUMN_CONTRACT in grp.columns and not grp[COLUMN_CONTRACT].isna().all() else "",
                "city": str(grp[COLUMN_CITY].dropna().iloc[0]) if COLUMN_CITY in grp.columns and not grp[COLUMN_CITY].isna().all() else "",
                "type_liv": str(grp[COLUMN_TYPE_LIV].dropna().iloc[0]) if COLUMN_TYPE_LIV in grp.columns and not grp[COLUMN_TYPE_LIV].isna().all() else "",
            }

        low = kpax_last.copy()
        low["serial_display"] = low["serial_display"].astype(str).str.strip()
        low["color"] = low["color"].astype(str).str.lower().str.strip()
        low["last_pct"] = pd.to_numeric(low["last_pct"], errors="coerce")
        low = low[low["last_pct"].notna() & (low["last_pct"] < 3)]

        extras = []
        if not low.empty:
            for _, row in low.iterrows():
                s = row["serial_display"]
                c = row["color"]

                # si deja recommande => rien
                if (s, c) in existing_pairs:
                    continue

                # lookup O(1) au lieu de df[df[...]==s]
                info = _serial_info.get(s, {})
                client_name = info.get("client", "")
                contract_no = info.get("contract", "")
                city = info.get("city", "")
                type_liv = info.get("type_liv", "")

                # Si pas de contrat => on ignore (pas de P4)
                if not contract_no or contract_no.strip() == "":
                    continue

                # Calculer days_since_last et rupture_kpax pour cette ligne P4
                last_update_val = row.get("last_update")
                if pd.notna(last_update_val):
                    last_update_dt = pd.to_datetime(last_update_val, errors="coerce")
                    if pd.notna(last_update_dt):
                        days_since = (today - last_update_dt).days
                        rupture_flag = days_since > KPAX_STALE_DAYS
                    else:
                        days_since = pd.NA
                        rupture_flag = True
                else:
                    days_since = pd.NA
                    rupture_flag = True

                extras.append(
                    {
                        COLUMN_SERIAL: s,
                        COLUMN_SERIAL_DISPLAY: s,
                        COLUMN_TONER: c,
                        "couleur": c,
                        "couleur_norm": c,
                        COLUMN_DAYS: pd.NA,
                        COLUMN_STOCKOUT: pd.NA,
                        COLUMN_PRIORITY: 4,
                        COLUMN_CLIENT: client_name,
                        COLUMN_CONTRACT: contract_no,
                        COLUMN_CITY: city,
                        COLUMN_TYPE_LIV: type_liv if type_liv else pd.NA,
                        COLUMN_COMMENT: "Toner bas mais non prioritaire",
                        COLUMN_LAST_UPDATE: last_update_val,
                        COLUMN_LAST_PCT: float(row["last_pct"]) if pd.notna(row["last_pct"]) else pd.NA,
                        "days_since_last": days_since,
                        "rupture_kpax": rupture_flag,
                        "warning_incoherence": False,
                        "fallback_p4": True,
                    }
                )

        if extras:
            extra_df = pd.DataFrame(extras)

            max_id = int(df[COLUMN_ID].max()) if COLUMN_ID in df.columns and df[COLUMN_ID].notna().any() else 0
            extra_df[COLUMN_ID] = range(max_id + 1, max_id + 1 + len(extra_df))

            df = pd.concat([df, extra_df], ignore_index=True)
            print(f"[P4] {len(extras)} toner(s) ajoute(s) en priorite 4 (< 3%, avec contrat)")

    # colonne fallback_p4 propre pour tout le DF
    if "fallback_p4" not in df.columns:
        df["fallback_p4"] = False
    else:
        df["fallback_p4"] = df["fallback_p4"].fillna(False).infer_objects(copy=False).astype(bool)

    # recast (securite)
    if COLUMN_PRIORITY in df.columns:
        df[COLUMN_PRIORITY] = pd.to_numeric(df[COLUMN_PRIORITY], errors="coerce").astype("Int64")
    if COLUMN_DAYS in df.columns:
        df[COLUMN_DAYS] = pd.to_numeric(df[COLUMN_DAYS], errors="coerce")

    return df


def get_data():
    """
    Cache de load_data() par mtime du fichier CSV source.
    Tant que le fichier n'a pas changé sur disque, on retourne
    une copie du DataFrame déjà calculé — pas de re-lecture, pas de re-merge.
    """
    global _DATA_CACHE

    if DATA_PATH.exists():
        mtime = DATA_PATH.stat().st_mtime
    else:
        mtime = 0

    if _DATA_CACHE is not None:
        cached_mtime, cached_df = _DATA_CACHE
        if cached_mtime == mtime:
            print("[DATA] Cache hit (mtime inchangé)")
            return cached_df.copy()

    print("[DATA] Cache miss → rechargement complet")
    df = _load_data_from_disk()
    _DATA_CACHE = (mtime, df)
    return df.copy()


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

    hist = load_kpax_history_for_serial(serial)
    sub = hist[hist["color"].astype(str).str.lower() == color].copy()

    # Si pas d'historique => fallback forecasts
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

    # ✅ lecture CHUNKED uniquement pour ce serial (safe Render)
    hist = load_kpax_history_for_serial(serial)

    series = {c: [] for c in ["black", "cyan", "magenta", "yellow"]}

    if not hist.empty:
        hist = hist.sort_values("date")
        for color, g in hist.groupby("color"):
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
    df = get_data()
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

    # ============================================================
    # FILTRER LES ALERTES NON PRIORITAIRES (contrats annulés avec date)
    # Ces lignes ne s'affichent PAS dans l'accueil
    # ============================================================
    if "alerte_non_prioritaire" in filtered.columns:
        # Convertir en bool proprement
        filtered["alerte_non_prioritaire"] = filtered["alerte_non_prioritaire"].fillna(False).infer_objects(copy=False).astype(bool)
        nb_alertes_total = filtered["alerte_non_prioritaire"].sum()
        filtered = filtered[~filtered["alerte_non_prioritaire"]].copy()
    else:
        nb_alertes_total = 0

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

            min_days_left = float(sub[COLUMN_DAYS].min()) if COLUMN_DAYS in sub.columns and sub[COLUMN_DAYS].notna().any() else None
            
            # Pour min_stockout_date, on filtre d'abord les NA avant de faire le min
            if COLUMN_STOCKOUT in sub.columns and sub[COLUMN_STOCKOUT].notna().any():
                min_stockout_date = str(sub[COLUMN_STOCKOUT].dropna().min())
            else:
                min_stockout_date = None
            
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

            # Alerte contrat fin proche
            alerte_fin_proche = False
            date_fin_contrat = None
            if "alerte_contrat_fin_proche" in sub.columns:
                alerte_fin_proche = sub["alerte_contrat_fin_proche"].any()
            if COLUMN_DATE_FIN_CONTRAT in sub.columns and sub[COLUMN_DATE_FIN_CONTRAT].notna().any():
                date_fin_contrat = sub[COLUMN_DATE_FIN_CONTRAT].dropna().iloc[0].strftime("%d/%m/%Y")

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
                "alerte_fin_proche": alerte_fin_proche,
                "date_fin_contrat": date_fin_contrat,
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
        nb_alertes_non_prioritaires=nb_alertes_total,
    )


@app.route("/alertes", methods=["GET"])
def alertes():
    """Page dédiée aux alertes non prioritaires (contrats annulés avec date)"""
    df = get_data()
    processed_ids = load_processed_ids()
    
    # Filtrer uniquement les alertes non prioritaires
    if "alerte_non_prioritaire" not in df.columns:
        df["alerte_non_prioritaire"] = False

    # fillna obligatoire : les lignes P4 ajoutées après le premier groupby
    # n'ont pas cette colonne → NA → ValueError sur le masque booléen.
    df["alerte_non_prioritaire"] = (
        df["alerte_non_prioritaire"].fillna(False).infer_objects(copy=False).astype(bool)
    )

    alertes_df = df[df["alerte_non_prioritaire"]].copy()
    
    # Trier
    sort_cols = []
    if COLUMN_DATE_FIN_CONTRAT in alertes_df.columns:
        sort_cols.append(COLUMN_DATE_FIN_CONTRAT)
    if COLUMN_PRIORITY in alertes_df.columns:
        sort_cols.append(COLUMN_PRIORITY)
    if COLUMN_DAYS in alertes_df.columns:
        sort_cols.append(COLUMN_DAYS)
    sort_cols.append(COLUMN_CLIENT)
    sort_cols.append(COLUMN_SERIAL_DISPLAY)
    
    if sort_cols:
        alertes_df = alertes_df.sort_values(sort_cols, na_position="last")
    
    # Grouper par imprimante
    grouped_printers = []
    if not alertes_df.empty:
        for serial_display, sub in alertes_df.groupby(COLUMN_SERIAL_DISPLAY):
            rows = sub.to_dict(orient="records")
            
            min_days_left = float(sub[COLUMN_DAYS].min()) if COLUMN_DAYS in sub.columns and sub[COLUMN_DAYS].notna().any() else None
            
            if COLUMN_STOCKOUT in sub.columns and sub[COLUMN_STOCKOUT].notna().any():
                min_stockout_date = str(sub[COLUMN_STOCKOUT].dropna().min())
            else:
                min_stockout_date = None
            
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
            
            # Date de fin du contrat
            date_fin_contrat = None
            if COLUMN_DATE_FIN_CONTRAT in sub.columns and sub[COLUMN_DATE_FIN_CONTRAT].notna().any():
                date_fin_contrat = sub[COLUMN_DATE_FIN_CONTRAT].dropna().iloc[0].strftime("%d/%m/%Y")
            
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
                "date_fin_contrat": date_fin_contrat,
            })
    
    return render_template(
        "alertes.html",
        grouped_printers=grouped_printers,
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
        total_alertes=len(alertes_df),
    )


@app.route("/printer/<serial_display>", methods=["GET"])
def printer_detail(serial_display):
    serial_display = (serial_display or "").strip()

    df = get_data()
    sub = df[df[COLUMN_SERIAL_DISPLAY].astype(str) == serial_display].copy()
    if sub.empty:
        abort(404)

    client_name = sub[COLUMN_CLIENT].dropna().astype(str).iloc[0] if COLUMN_CLIENT in sub.columns and not sub[COLUMN_CLIENT].isna().all() else ""
    contract_no = sub[COLUMN_CONTRACT].dropna().astype(str).iloc[0] if COLUMN_CONTRACT in sub.columns and not sub[COLUMN_CONTRACT].isna().all() else ""
    city = sub[COLUMN_CITY].dropna().astype(str).iloc[0] if COLUMN_CITY in sub.columns and not sub[COLUMN_CITY].isna().all() else ""

    min_days_left = sub[COLUMN_DAYS].min() if COLUMN_DAYS in sub.columns and sub[COLUMN_DAYS].notna().any() else None
    
    if COLUMN_STOCKOUT in sub.columns and sub[COLUMN_STOCKOUT].notna().any():
        min_stockout_date = sub[COLUMN_STOCKOUT].dropna().min()
    else:
        min_stockout_date = None
    
    min_priority = sub[COLUMN_PRIORITY].min() if COLUMN_PRIORITY in sub.columns and sub[COLUMN_PRIORITY].notna().any() else None

    rows = sub.to_dict(orient="records")

    slopes_map = get_slopes_map()
    slopes = {c: slopes_map.get((serial_display, c)) for c in ["black", "cyan", "magenta", "yellow"]}

    # KPAX status depuis le CSV léger "kpax_last_states.csv"
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