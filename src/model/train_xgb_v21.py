from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


DATA_ML = Path("data/processed/ml")
MODEL_DIR = Path("models")


def main():
    # 🔥 CHANGEMENT : on utilise le dataset avec les VRAIS labels
    path = DATA_ML / "training_labels_v21.parquet"
    
    if not path.exists():
        print(f"[train_xgb_v21] ERREUR: fichier introuvable: {path}")
        print("[train_xgb_v21] Exécute d'abord: python -m src.model.compute_ml_training_labels")
        raise SystemExit(1)

    df = pd.read_parquet(path)
    print(f"[train_xgb_v21] dataset_v21 shape = {df.shape}")

    # --- 1) Vérif présence de la cible (VRAIS OFFSETS) ---
    if "offset_days_real" not in df.columns:
        raise SystemExit(
            "[train_xgb_v21] ERREUR: colonne 'offset_days_real' absente. "
            "Vérifie compute_ml_training_labels.py."
        )

    # On enlève les lignes sans cible
    df = df.dropna(subset=["offset_days_real"]).copy()
    if df.empty:
        raise SystemExit("[train_xgb_v21] ERREUR: aucune ligne avec offset_days_real non nul.")

    print(f"[train_xgb_v21] {len(df)} exemples d'entraînement avec vrais labels")

    # --- 2) Cible & features ---
    y = df["offset_days_real"].astype(float)

    # On garde uniquement les colonnes numériques
    num_df = df.select_dtypes(include=[np.number]).copy()

    # On retire la cible + colonnes liées aux dates réelles (pour éviter le data leakage)
    cols_to_drop = ["offset_days_real", "ship_date_real"]
    
    # On retire aussi les anciennes colonnes d'offset si présentes
    if "offset_days" in num_df.columns:
        cols_to_drop.append("offset_days")
    
    X = num_df.drop(columns=[c for c in cols_to_drop if c in num_df.columns])

    feature_cols = list(X.columns)
    print(f"[train_xgb_v21] nb features numériques = {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise SystemExit("[train_xgb_v21] ERREUR: aucune feature numérique disponible.")

    # --- 3) Split train / test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 4) Modèle XGBoost V2.1 ---
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        tree_method="hist",
    )

    print("[train_xgb_v21] training sur les VRAIS offsets...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- 5) Évaluation ---
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("===== XGB Offset Model V2.1 (VRAIS LABELS) =====")
    print(f"MAE   : {mae:.2f} jours")
    print(f"RMSE  : {rmse:.2f} jours")
    print(f"R²    : {r2:.3f}")
    
    # Stats sur les offsets réels
    print(f"\nStats offsets réels:")
    print(f"  Moyenne : {y.mean():.1f} jours")
    print(f"  Médiane : {y.median():.1f} jours")
    print(f"  Écart-type : {y.std():.1f} jours")

    # --- 6) Sauvegarde ---
    MODEL_DIR.mkdir(exist_ok=True)

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
    }

    out_path = MODEL_DIR / "xgb_offset_model_v21.pkl"
    joblib.dump(bundle, out_path)
    print(f"[train_xgb_v21] modèle sauvegardé → {out_path}")


if __name__ == "__main__":
    main()