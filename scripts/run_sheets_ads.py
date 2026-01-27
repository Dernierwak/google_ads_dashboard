import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from meta.ads.sheets import send_ads_facebook_to_google_sheet


if __name__ == "__main__":
    send_ads_facebook_to_google_sheet()
