import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from meta.ads.export import transform_json_fb_ads_insight_to_excel


if __name__ == "__main__":
    transform_json_fb_ads_insight_to_excel()
