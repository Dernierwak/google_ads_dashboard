from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
ADS_DIR = DATA_DIR / "ads"
ADS_JSON_DIR = ADS_DIR / "json"
ADS_EXCEL_DIR = ADS_DIR / "excel"

ORGANIC_DIR = DATA_DIR / "organic"
ORGANIC_FB_JSON_DIR = ORGANIC_DIR / "fb" / "json"
ORGANIC_FB_EXCEL_DIR = ORGANIC_DIR / "fb" / "excel"
ORGANIC_IG_JSON_DIR = ORGANIC_DIR / "ig" / "json"
ORGANIC_IG_EXCEL_DIR = ORGANIC_DIR / "ig" / "excel"
ORGANIC_STORIES_CSV_DIR = ORGANIC_DIR / "stories" / "csv"

CONNECTION_DIR = ROOT / "connection_token"
SERVICE_ACCOUNT_FILE = CONNECTION_DIR / "service-account.json"

