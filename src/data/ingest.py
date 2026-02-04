from pathlib import Path
import pandas as pd
import re

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

def find_latest(pattern: str) -> Path:
    candidates = list(RAW.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern: {pattern}")
    
    def extract_date(p: Path):
        # cherche un bloc de 6, 7 ou 8 chiffres type 422026, 3012026, 30012026
        m = re.search(r"(\d{6,8})", p.name)
        if not m:
            return None
        s = m.group(1)
        
        if len(s) == 8:
            # Format DDMMYYYY (ex: 30012026)
            dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
        elif len(s) == 7:
            # Format DDMYYYY (ex: 3012026 = 30/1/2026)
            dd = int(s[:2])
            mm = int(s[2])      # 1 chiffre pour le mois
            yyyy = int(s[3:])
        elif len(s) == 6:
            # Format DMYYYY (ex: 422026 = 4/2/2026)
            dd = int(s[0])      # 1 chiffre pour le jour
            mm = int(s[1])      # 1 chiffre pour le mois
            yyyy = int(s[2:])
        else:
            return None
        
        return (yyyy, mm, dd)
    
    dated = [(p, extract_date(p)) for p in candidates]
    dated_ok = [(p, d) for p, d in dated if d is not None]
    
    if dated_ok:
        latest = max(dated_ok, key=lambda t: t[1])[0]
    else:
        # fallback si aucun nom n'a de date : on reprend mtime
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
    
    print(f"[ingest] latest for {pattern}: {latest.name}")
    return latest

def load_kpax_consumables_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*Kpax*consumables*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df

def load_item_ledger_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*Item*Ledg*Entries*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df

def load_meters_adex() -> pd.DataFrame:
    path = find_latest("AI*Export*SalesPages*Meters*ADEXGROUP*")
    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8", engine="python")
    return df

def main():
    # 1) Item Ledger
    item_ledger = load_item_ledger_adex()
    out_item = INTERIM / "item_ledger.csv"
    item_ledger.to_csv(out_item, index=False)
    print(f"Saved: {out_item}")
    
    # 2) KPAX consumables
    kpax = load_kpax_consumables_adex()
    out_kpax = INTERIM / "kpax_consumables.csv"
    kpax.to_csv(out_kpax, index=False)
    print(f"Saved: {out_kpax}")
    
    # 3) Meters
    meters = load_meters_adex()
    out_meters = INTERIM / "meters.csv"
    meters.to_csv(out_meters, index=False)
    print(f"Saved: {out_meters}")

if __name__ == "__main__":
    main()