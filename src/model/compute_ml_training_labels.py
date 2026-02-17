"""
src/model/compute_ml_training_labels.py

Calcule les VRAIS labels d'entraînement pour le modèle ML V2.1
en comparant les anciennes prédictions avec les livraisons réelles.

Ce script crée un dataset avec :
- Les features des prédictions passées
- Les VRAIS offsets calculés depuis les livraisons réelles
  (offset_réel = date_livraison_réelle - date_rupture_prédite)

Usage:
    python -m src.model.compute_ml_training_labels
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Réutiliser les fonctions de feature engineering
from src.model.build_dataset_v21 import (
    feat_time_features,
    feat_cycle_stats,
    feat_usage_speed,
    feat_meters,
)


DATA_PROCESSED = Path("data/processed")
ML_DATA = DATA_PROCESSED / "ml"
OUTPUT_PATH = ML_DATA / "training_labels_v21.parquet"


def normalize_serial(s):
    """Normalise un numéro de série"""
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def load_historical_forecasts() -> pd.DataFrame:
    """
    Charge les prédictions historiques (slopes V1).
    On part de consumables_forecasts.parquet qui contient l'historique.
    """
    path = DATA_PROCESSED / "consumables_forecasts.parquet"
    if not path.exists():
        print(f"[training_labels] WARN: {path} introuvable")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    
    # Normaliser serial
    if "serial_norm" not in df.columns:
        df["serial_norm"] = df["serial"].apply(normalize_serial)
    else:
        df["serial_norm"] = df["serial_norm"].apply(normalize_serial)
    
    # S'assurer que les dates sont en datetime
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")
    df["stockout_date"] = pd.to_datetime(df["stockout_date"], errors="coerce")
    
    return df


def load_ledger_shipments() -> pd.DataFrame:
    """
    Charge les livraisons réelles depuis item_ledger.
    """
    path = DATA_PROCESSED / "item_ledger.parquet"
    if not path.exists():
        print(f"[training_labels] WARN: {path} introuvable")
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    
    # Trouver les colonnes
    serial_col = None
    for col in ["serial", "serial_norm", "$No. serie$", "No. serie"]:
        if col in df.columns:
            serial_col = col
            break
    
    date_col = None
    for col in ["doc_date", "doc_datetime", "$Date compta$", "Date compta"]:
        if col in df.columns:
            date_col = col
            break
    
    type_col = None
    for col in ["consumable_type", "$Type conso$", "Type conso"]:
        if col in df.columns:
            type_col = col
            break
    
    if serial_col is None or date_col is None:
        print(f"[training_labels] WARN: colonnes manquantes dans ledger")
        return pd.DataFrame()
    
    # Créer un dataframe propre
    ledger = pd.DataFrame()
    ledger["serial_norm"] = df[serial_col].apply(normalize_serial)
    ledger["ship_date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    if type_col:
        # Extraire la couleur
        ledger["consumable_type"] = df[type_col].astype(str)
        
        def extract_color(type_str):
            s = str(type_str).lower()
            if "noir" in s or "black" in s:
                return "black"
            elif "cyan" in s:
                return "cyan"
            elif "magenta" in s:
                return "magenta"
            elif "jaune" in s or "yellow" in s:
                return "yellow"
            return None
        
        ledger["color"] = ledger["consumable_type"].apply(extract_color)
    
    # Ne garder que les livraisons valides
    ledger = ledger.dropna(subset=["serial_norm", "ship_date"])
    
    return ledger


def match_predictions_with_shipments(forecasts: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Matche les prédictions avec les livraisons réelles.
    
    Logique :
    - Pour chaque prédiction (serial + color + stockout_date)
    - Chercher une livraison réelle dans une fenêtre temporelle
    - Calculer l'offset réel
    """
    
    if forecasts.empty or ledger.empty:
        print("[training_labels] Forecasts ou ledger vide")
        return pd.DataFrame()
    
    # Ne garder que les prédictions avec date de rupture valide
    forecasts = forecasts[forecasts["stockout_date"].notna()].copy()
    
    matched_rows = []
    
    print(f"[training_labels] Matching {len(forecasts)} prédictions avec {len(ledger)} livraisons...")
    
    for idx, pred in forecasts.iterrows():
        serial = pred["serial_norm"]
        color = str(pred.get("color", "")).lower()
        stockout_pred = pred["stockout_date"]
        
        if pd.isna(serial) or pd.isna(stockout_pred):
            continue
        
        # Chercher les livraisons pour ce serial + couleur
        # dans une fenêtre de ±120 jours autour de la date prédite
        window_start = stockout_pred - pd.Timedelta(days=60)
        window_end = stockout_pred + pd.Timedelta(days=120)
        
        mask = (
            (ledger["serial_norm"] == serial) &
            (ledger["ship_date"] >= window_start) &
            (ledger["ship_date"] <= window_end)
        )
        
        if "color" in ledger.columns:
            mask = mask & (ledger["color"] == color)
        
        matches = ledger[mask]
        
        if matches.empty:
            continue
        
        # Prendre la livraison la plus proche de la date prédite
        matches = matches.copy()
        matches["date_diff"] = (matches["ship_date"] - stockout_pred).abs()
        best_match = matches.loc[matches["date_diff"].idxmin()]
        
        # Calculer l'offset réel (en jours)
        real_offset = (best_match["ship_date"] - stockout_pred).days
        
        # Créer une ligne avec la prédiction + le vrai offset
        row = pred.to_dict()
        row["ship_date_real"] = best_match["ship_date"]
        row["offset_days_real"] = int(real_offset)
        row["matched"] = True
        
        matched_rows.append(row)
    
    if not matched_rows:
        print("[training_labels] Aucun match trouvé")
        return pd.DataFrame()
    
    result = pd.DataFrame(matched_rows)
    print(f"[training_labels] {len(result)} prédictions matchées avec des livraisons réelles")
    
    return result


def add_features_to_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features avancées au dataset d'entraînement
    (même logique que build_dataset_v21)
    """
    
    # Charger les sources de features
    resets_path = DATA_PROCESSED / "consumables_with_resets.parquet"
    meters_path = DATA_PROCESSED / "meters.parquet"
    
    if resets_path.exists():
        resets = pd.read_parquet(resets_path)
    else:
        resets = pd.DataFrame()
    
    if meters_path.exists():
        meters = pd.read_parquet(meters_path)
    else:
        meters = pd.DataFrame()
    
    # Time features
    time_feats = feat_time_features(df)
    df = df.merge(time_feats, on=["serial_norm", "color"], how="left")
    
    # Cycle stats
    if not resets.empty:
        cycle_stats = feat_cycle_stats(resets)
        if not cycle_stats.empty:
            df = df.merge(cycle_stats, on=["serial_norm", "color"], how="left")
    
    # Usage speed
    if not resets.empty:
        usage = feat_usage_speed(resets)
        if not usage.empty:
            df = df.merge(usage, on=["serial_norm", "color"], how="left")
    
    # Meters
    if not meters.empty:
        mt_feats = feat_meters(meters)
        if not mt_feats.empty:
            df = df.merge(mt_feats, on="serial_norm", how="left")
    
    return df


def main():
    """
    Pipeline principal :
    1. Charge les prédictions historiques
    2. Charge les livraisons réelles
    3. Matche les deux pour calculer les vrais offsets
    4. Ajoute les features avancées
    5. Sauvegarde le dataset d'entraînement
    """
    
    print("[training_labels] Chargement des données...")
    
    # 1) Prédictions historiques
    forecasts = load_historical_forecasts()
    if forecasts.empty:
        print("[training_labels] Aucune prédiction historique → dataset vide")
        ML_DATA.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(OUTPUT_PATH, index=False)
        return
    
    print(f"[training_labels] {len(forecasts)} prédictions chargées")
    
    # 2) Livraisons réelles
    ledger = load_ledger_shipments()
    if ledger.empty:
        print("[training_labels] Aucune livraison réelle → dataset vide")
        ML_DATA.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(OUTPUT_PATH, index=False)
        return
    
    print(f"[training_labels] {len(ledger)} livraisons chargées")
    
    # 3) Matching
    training_data = match_predictions_with_shipments(forecasts, ledger)
    
    if training_data.empty:
        print("[training_labels] Aucun match → dataset vide")
        ML_DATA.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(OUTPUT_PATH, index=False)
        return
    
    # 4) Ajout des features avancées
    print("[training_labels] Ajout des features avancées...")
    training_data = add_features_to_training_data(training_data)
    
    # 5) Nettoyage final
    # Remplacer les inf/nan dans les features numériques
    numeric_cols = training_data.select_dtypes(include=[np.number]).columns
    training_data[numeric_cols] = training_data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 6) Sauvegarde
    ML_DATA.mkdir(parents=True, exist_ok=True)
    training_data.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"[training_labels] Dataset d'entraînement sauvegardé → {OUTPUT_PATH}")
    print(f"[training_labels] {len(training_data)} exemples avec vrais labels")
    
    # Stats utiles
    if len(training_data) > 0:
        print(f"[training_labels] Offset réel moyen : {training_data['offset_days_real'].mean():.1f} jours")
        print(f"[training_labels] Offset réel médian : {training_data['offset_days_real'].median():.1f} jours")
        print(f"[training_labels] Min/Max : {training_data['offset_days_real'].min()}/{training_data['offset_days_real'].max()} jours")


if __name__ == "__main__":
    main()