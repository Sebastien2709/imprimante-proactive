"""
src/data/load_contract_status.py

Charge les statuts de contrats depuis AI_Export_ItemLedgEntries_*.txt
et crée un fichier processed : contract_status.parquet

Ce fichier contient :
- serial_norm : numéro de série normalisé
- statut_contrat : Signé / Annulé
- date_fin_contrat : date de fin prévue du contrat

Usage:
    python -m src.data.load_contract_status
    ou
    python src/data/load_contract_status.py
"""

from pathlib import Path
import pandas as pd
import re


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "contract_status.parquet"


def norm_col(s: str) -> str:
    """
    Normalise un nom de colonne pour faciliter le matching.
    Même logique que dans build_contracted_devices.py
    """
    if not isinstance(s, str):
        return ""
    
    repl = str(s).lower().strip()
    
    # Enlève les caractères spéciaux
    for ch in [" ", ".", "_", "-", "'", "$"]:
        repl = repl.replace(ch, "")
    
    # Enlève les accents
    return (
        repl.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
    )


def pick_col(df: pd.DataFrame, candidates: list[str]):
    """
    Trouve la première colonne correspondant à l'un des candidats.
    Même logique que dans build_contracted_devices.py
    """
    norm_map = {norm_col(c): c for c in df.columns}
    
    # 1er passage : match exact sur norm()
    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    
    # 2ème passage : match "contient" (plus flou)
    for c in df.columns:
        nc = norm_col(c)
        for cand in candidates:
            if norm_col(cand) in nc:
                return c
    
    return None


def find_latest_item_ledger():
    """
    Trouve le fichier AI_Export_ItemLedgEntries_*.txt le plus récent.
    Même logique que dans ingest.py
    """
    pattern = "AI*Export*Item*Ledg*Entries*ADEXGROUP*"
    candidates = list(RAW_DIR.glob(pattern))
    
    if not candidates:
        print(f"[load_contract_status] Aucun fichier trouvé pour : {pattern}")
        return None
    
    def extract_date(p: Path):
        """Extrait la date du nom de fichier (format DDMMYYYY ou DMMYYYY)"""
        m = re.search(r"(\d{7,8})", p.name)
        if not m:
            return None
        s = m.group(1)
        
        if len(s) == 8:
            # Format DDMMYYYY (ex: 30012026)
            dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
        else:
            # Format 7 chiffres DMMYYYY (ex: 3012026 = 30/1/2026)
            dd = int(s[:2])
            mm = int(s[2:-4])
            yyyy = int(s[-4:])
        
        return (yyyy, mm, dd)
    
    dated = [(p, extract_date(p)) for p in candidates]
    dated_ok = [(p, d) for p, d in dated if d is not None]
    
    if dated_ok:
        latest = max(dated_ok, key=lambda t: t[1])[0]
    else:
        # Fallback : mtime
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
    
    return latest


def normalize_serial(s) -> str:
    """Normalise un numéro de série : upper + strip"""
    if pd.isna(s):
        return ""
    return str(s).strip().upper().replace("$", "")


def load_contract_status() -> pd.DataFrame:
    """
    Charge le fichier ItemLedgEntries et extrait :
    - serial_norm
    - statut_contrat (Signé / Annulé)
    - date_fin_contrat
    """
    
    latest_file = find_latest_item_ledger()
    
    if latest_file is None:
        print("[load_contract_status] Aucun fichier ItemLedgEntries trouvé → retour DataFrame vide")
        return pd.DataFrame(columns=["serial_norm", "statut_contrat", "date_fin_contrat"])
    
    print(f"[load_contract_status] Lecture : {latest_file.name}")
    
    try:
        # Lecture avec pipe separator
        df = pd.read_csv(latest_file, sep="|", dtype=str, encoding="utf-8", low_memory=False)
        
        print(f"[load_contract_status] {len(df)} lignes chargées")
        print(f"[load_contract_status] Colonnes disponibles (10 premières) : {list(df.columns[:10])}")
        
        # --- Détection des colonnes ---
        
        # 1) Numéro de série
        serial_col = pick_col(df, [
            "No. serie", "No serie", "serial", "$No serie$", "$No. serie$",
            "no_serie", "NoSerie", "Serial"
        ])
        
        if serial_col is None:
            print("[load_contract_status] Colonne serial introuvable")
            return pd.DataFrame(columns=["serial_norm", "statut_contrat", "date_fin_contrat"])
        
        print(f"[load_contract_status] Colonne serial détectée : {serial_col}")
        
        # 2) Statut contrat
        statut_col = pick_col(df, [
            "Statut contrat", "$Statut contrat$", "statut_contrat",
            "Contract Status", "ContractStatus"
        ])
        
        if statut_col is None:
            print("[load_contract_status] Colonne 'Statut contrat' introuvable")
            print(f"[load_contract_status] Colonnes disponibles : {list(df.columns)}")
            return pd.DataFrame(columns=["serial_norm", "statut_contrat", "date_fin_contrat"])
        
        print(f"[load_contract_status] Colonne statut détectée : {statut_col}")
        
        # 3) Date fin contrat
        date_fin_col = pick_col(df, [
            "Date fin contrat", "$Date fin contrat$", "date_fin_contrat",
            "Contract End Date", "DateFinContrat"
        ])
        
        if date_fin_col is None:
            print("[load_contract_status] Colonne 'Date fin contrat' introuvable")
            return pd.DataFrame(columns=["serial_norm", "statut_contrat", "date_fin_contrat"])
        
        print(f"[load_contract_status] Colonne date fin détectée : {date_fin_col}")
        
        # --- Extraction et nettoyage ---
        
        contracts = df[[serial_col, statut_col, date_fin_col]].copy()
        
        # Normalisation du serial
        contracts["serial_norm"] = contracts[serial_col].apply(normalize_serial)
        
        # Nettoyage du statut
        contracts["statut_contrat"] = (
            contracts[statut_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        
        # Nettoyage et parsing de la date
        date_raw = (
            contracts[date_fin_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        
        # Parsing avec plusieurs formats possibles
        date_parsed = pd.to_datetime(date_raw, format="%d/%m/%y", errors="coerce")
        
        # Fallback si trop de NaT
        if date_parsed.isna().mean() > 0.9:
            date_parsed = pd.to_datetime(date_raw, format="%d/%m/%Y", errors="coerce")
        
        contracts["date_fin_contrat"] = date_parsed
        
        # --- Nettoyage final ---
        
        # On garde uniquement les lignes avec serial valide
        contracts = contracts[
            contracts["serial_norm"].notna() & 
            (contracts["serial_norm"] != "")
        ].copy()
        
        # Dédupliquer : garder la dernière entrée par serial
        contracts = contracts.drop_duplicates(subset=["serial_norm"], keep="last")
        
        # Colonnes finales
        contracts = contracts[["serial_norm", "statut_contrat", "date_fin_contrat"]]
        
        print(f"[load_contract_status] {len(contracts)} contrats uniques extraits")
        
        # Stats utiles
        if len(contracts) > 0:
            statut_counts = contracts["statut_contrat"].value_counts().to_dict()
            print(f"[load_contract_status] Statuts : {statut_counts}")
            
            nb_with_date = contracts["date_fin_contrat"].notna().sum()
            print(f"[load_contract_status] Contrats avec date fin : {nb_with_date}")
        
        return contracts
        
    except Exception as e:
        print(f"[load_contract_status] Erreur lors du chargement : {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["serial_norm", "statut_contrat", "date_fin_contrat"])


def main():
    """Charge et sauvegarde le statut des contrats"""
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    contracts = load_contract_status()
    
    if contracts.empty:
        print("[load_contract_status] Aucun contrat chargé → fichier vide créé")
    
    contracts.to_parquet(OUTPUT_FILE, index=False)
    print(f"[load_contract_status] Sauvegardé → {OUTPUT_FILE} ({len(contracts)} lignes)")


if __name__ == "__main__":
    main()