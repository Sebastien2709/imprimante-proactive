import pandas as pd
from pathlib import Path
import glob

# Chercher le fichier SalesPagesMeters
files = glob.glob("data/raw/AI_Export_SalesPages*Meters*.txt")
if files:
    latest = max(files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Fichier trouvé : {latest}")
    
    df = pd.read_csv(latest, sep="|", nrows=5, encoding="utf-8", low_memory=False)
    
    print("\n=== COLONNES ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
else:
    print("Aucun fichier SalesPagesMeters trouvé")