import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from meta.organic.export import main_fb, main_insta


if __name__ == "__main__":
    main_fb()
    main_insta()
