import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ORGANIC_DIR = ROOT / "organic_performance"
if str(ORGANIC_DIR) not in sys.path:
    sys.path.append(str(ORGANIC_DIR))

from organic_performance.main_streamlit import main


if __name__ == "__main__":
    main()
