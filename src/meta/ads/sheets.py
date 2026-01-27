import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from core.paths import ADS_DIR, ADS_EXCEL_DIR, SERVICE_ACCOUNT_FILE

for directory in [ADS_DIR, ADS_EXCEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def send_ads_facebook_to_google_sheet() -> None:
    df_ads_insights = pd.read_excel(ADS_EXCEL_DIR / "final_ads_insights.xlsx")

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_FILE), scopes=scope)
    client = gspread.authorize(creds)

    sheet = client.open("Facebook Ads - Performances").worksheet("automatisation - py")
    sheet.clear()
    sheet.update([df_ads_insights.columns.values.tolist()] + df_ads_insights.values.tolist(), value_input_option="USER_ENTERED")


if __name__ == "__main__":
    send_ads_facebook_to_google_sheet()
