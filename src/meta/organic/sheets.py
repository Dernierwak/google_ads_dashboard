import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from core.paths import ORGANIC_FB_EXCEL_DIR, ORGANIC_IG_EXCEL_DIR, SERVICE_ACCOUNT_FILE


def send_insta_post_to_google_sheet() -> None:
    df_post_insta = pd.read_excel(ORGANIC_IG_EXCEL_DIR / "insight_per_media_insta.xlsx")
    df_post_insta["date"] = pd.to_datetime(df_post_insta["date"]).apply(lambda x: x.date().isoformat())

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_FILE), scopes=scope)
    client = gspread.authorize(creds)

    sheet = client.open("Instagram - Org Performances").worksheet("automatisation - insta")
    sheet.clear()
    sheet.update([df_post_insta.columns.values.tolist()] + df_post_insta.values.tolist(), value_input_option="USER_ENTERED")


def send_insta_page_insights_to_google_sheet(spreadsheet_name="Instagram - Page Insights", worksheet_name="automatisation - page"):
    file_path = ORGANIC_IG_EXCEL_DIR / "ig_page_insights.xlsx"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df_page_insta = pd.read_excel(file_path)
    if "date" in df_page_insta.columns:
        df_page_insta["date"] = pd.to_datetime(df_page_insta["date"]).apply(lambda x: x.date().isoformat())

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_FILE), scopes=scope)
    client = gspread.authorize(creds)

    try:
        sheet = client.open(spreadsheet_name).worksheet(worksheet_name)
    except gspread.SpreadsheetNotFound:
        print(f"Spreadsheet '{spreadsheet_name}' not found")
        return
    except gspread.WorksheetNotFound:
        print(f"Worksheet '{worksheet_name}' not found in '{spreadsheet_name}'")
        return

    sheet.clear()
    sheet.update([df_page_insta.columns.values.tolist()] + df_page_insta.values.tolist(), value_input_option="USER_ENTERED")


def send_facebook_to_google_sheet(google_sheet_list: list | str):
    df_page_facebook_insights = pd.read_excel(ORGANIC_FB_EXCEL_DIR / "fb_page_insights.xlsx")
    df_post_facebook_insights = pd.read_excel(ORGANIC_FB_EXCEL_DIR / "fb_merge_insights_metrics.xlsx")

    valid_sheets = {
        "Page Insights": {
            "spreadsheet_name": "Facebook - Page Insights",
            "worksheet_name": "automatisation - py",
            "dataframe": df_page_facebook_insights,
        },
        "Post Insights": {
            "spreadsheet_name": "Facebook - Post Insights",
            "worksheet_name": "automatisation - py",
            "dataframe": df_post_facebook_insights,
        },
    }

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(str(SERVICE_ACCOUNT_FILE), scopes=scope)
    client = gspread.authorize(creds)

    if isinstance(google_sheet_list, str):
        google_sheet_list = [google_sheet_list]

    for sheet_name in google_sheet_list:
        if sheet_name not in valid_sheets:
            raise ValueError(f"Invalid sheet name '{sheet_name}'")

        config = valid_sheets[sheet_name]
        sheet = client.open(config["spreadsheet_name"]).worksheet(config["worksheet_name"])
        sheet.clear()
        sheet.update([config["dataframe"].columns.tolist()] + config["dataframe"].values.tolist(), value_input_option="USER_ENTERED")


if __name__ == "__main__":
    send_insta_post_to_google_sheet()
    send_insta_page_insights_to_google_sheet()
    send_facebook_to_google_sheet(google_sheet_list=["Post Insights", "Page Insights"])
