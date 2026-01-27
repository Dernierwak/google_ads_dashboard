import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
CAMPAIGNS_DIR = ROOT / "campaigns"
if str(CAMPAIGNS_DIR) not in sys.path:
    sys.path.append(str(CAMPAIGNS_DIR))

from campaigns.streamlit_ads import main


if __name__ == "__main__":
    main()
