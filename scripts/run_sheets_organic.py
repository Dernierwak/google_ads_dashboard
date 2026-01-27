import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from meta.organic.sheets import send_facebook_to_google_sheet, send_insta_page_insights_to_google_sheet, send_insta_post_to_google_sheet


if __name__ == "__main__":
    send_insta_post_to_google_sheet()
    send_insta_page_insights_to_google_sheet()
    send_facebook_to_google_sheet(google_sheet_list=["Post Insights", "Page Insights"])
