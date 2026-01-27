import json
import os

import pandas as pd

from core.paths import (
    ORGANIC_FB_EXCEL_DIR,
    ORGANIC_FB_JSON_DIR,
    ORGANIC_IG_EXCEL_DIR,
    ORGANIC_IG_JSON_DIR,
)
from .sheets import (
    send_facebook_to_google_sheet,
    send_insta_page_insights_to_google_sheet,
    send_insta_post_to_google_sheet,
)
from .stories import extract_stories_excel

pd.set_option("display.max_columns", None)


# FB

def convert_page_json_insight_to_excel(folder=ORGANIC_FB_JSON_DIR):
    ORGANIC_FB_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

    filename = "fb_page_insights.json"
    path_file = folder / filename
    if not path_file.exists():
        print(f"File not found: {path_file}")
        return

    with open(path_file, "r", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            print(f"{filename} is empty")
            return
        datas = json.loads(content)

    impressions_unique = []
    views_total = []
    impressions_organic = []
    dates = []
    page_daily_follows = []
    page_fan_adds_by_paid_non_paid_unique = []
    page_follows = []
    page_daily_unfollows_unique = []

    for data in datas:
        metric_name = data.get("name")
        values = data.get("values", [])
        for value in values:
            if metric_name == "page_impressions_unique":
                impressions_unique.append(value.get("value"))
                dates.append(value.get("end_time"))
            elif metric_name == "page_views_total":
                views_total.append(value.get("value"))
            elif metric_name == "page_posts_impressions_organic":
                impressions_organic.append(value.get("value"))
            elif metric_name == "page_daily_follows":
                page_daily_follows.append(value.get("value"))
            elif metric_name == "page_fan_adds_by_paid_non_paid_unique":
                page_fan_adds_by_paid_non_paid_unique.append(value.get("value"))
            elif metric_name == "page_follows":
                page_follows.append(value.get("value"))
            elif metric_name == "page_daily_unfollows_unique":
                page_daily_unfollows_unique.append(value.get("value"))

    structure_data = {
        "id": "1436103386609113",
        "page_impressions_unique": impressions_unique,
        "page_views_total": views_total,
        "page_posts_impressions_organic": impressions_organic,
        "date": dates,
        "page_daily_follows": page_daily_follows,
        "page_follows": page_follows,
        "page_daily_unfollows_unique": page_daily_unfollows_unique,
        "page_fan_adds_by_paid_non_paid_unique_total": [total.get("total") for total in page_fan_adds_by_paid_non_paid_unique],
        "page_fan_adds_by_paid_non_paid_unique_paid": [paid.get("paid") for paid in page_fan_adds_by_paid_non_paid_unique],
        "page_fan_adds_by_paid_non_paid_unique_unpaid": [unpaid.get("unpaid") for unpaid in page_fan_adds_by_paid_non_paid_unique],
    }

    df_structure = pd.DataFrame(structure_data)
    df_structure["all_date"] = df_structure["date"]
    df_structure["hours"] = pd.to_datetime(df_structure["date"]).dt.strftime("%H:%M:%S")
    df_structure["date"] = pd.to_datetime(df_structure["date"]).dt.strftime("%Y/%m/%d")
    df_structure = df_structure.astype({
        "page_follows": "Int64",
        "page_impressions_unique": "Int64",
        "page_views_total": "Int64",
        "page_posts_impressions_organic": "Int64",
    })
    df_structure["id"] = "'" + df_structure["id"]
    df_structure.drop(columns="all_date", inplace=True)
    df_structure.to_excel(ORGANIC_FB_EXCEL_DIR / "fb_page_insights.xlsx", index=False)


def convert_post_insight_json_to_excel(folder=ORGANIC_FB_JSON_DIR):
    ORGANIC_FB_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

    filename = "fb_posts_insights_2025.json"
    path_file = folder / filename
    if not path_file.exists():
        print(f"File not found: {path_file}")
        return

    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for post in data:
        post_id = post.get("post_id")
        created_time = post.get("created_time")

        for insight in post.get("insights", []):
            name = insight.get("name")
            for entry in insight.get("values", []):
                value = entry.get("value")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        records.append({
                            "post_id": post_id,
                            "created_time": created_time,
                            "metric": f"{name}_{sub_key}",
                            "value": sub_value,
                        })
                else:
                    records.append({
                        "post_id": post_id,
                        "created_time": created_time,
                        "metric": name,
                        "value": value,
                    })

    df_post_insights = pd.DataFrame(records)

    if not df_post_insights.empty:
        df_post_insights["date"] = pd.to_datetime(df_post_insights["created_time"]).dt.strftime("%Y.%m.%d")
        df_post_insights["hour"] = pd.to_datetime(df_post_insights["created_time"]).dt.strftime("%H:%M:%S")
        df_post_insights_copy = df_post_insights.copy()

        df_pivot = df_post_insights_copy.pivot_table(
            columns="metric",
            index=["post_id", "created_time", "date", "hour"],
            values="value",
            aggfunc="max",
        )

        df_pivot.reset_index(inplace=True)
        df_pivot.fillna(0, inplace=True)
        df_pivot.drop(columns=["post_reactions_like_total", "created_time"], inplace=True)
        df_pivot.to_excel(ORGANIC_FB_EXCEL_DIR / "fb_post_insights.xlsx", index=False)
    else:
        print("No data to export")


def convert_post_metrics_json_to_excel(folder=ORGANIC_FB_JSON_DIR):
    ORGANIC_FB_EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    filename = "fb_posts_metrics_2025.json"
    path_file = folder / filename

    if path_file.exists():
        with open(path_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for post in data:
            comments_data = post.get("comments_data", {}).get("data", []) if isinstance(post.get("comments_data"), dict) else []

            commentaire_p = [m.get("message") for m in comments_data if isinstance(m, dict) and "message" in m]

            commentaire_p_sous_commentaire = []
            for m in comments_data:
                if isinstance(m, dict) and m.get("comment_count", 0) >= 1:
                    replies = m.get("comments", {}).get("data", [])
                    if replies:
                        for x in replies:
                            if isinstance(x, dict) and "message" in x:
                                commentaire_p_sous_commentaire.append({
                                    "commentaire_principal": m.get("message", ""),
                                    "sous_commentaires": x.get("message", ""),
                                })

            records.append({
                "post_id": post.get("post_id"),
                "created_time": post.get("created_time"),
                "share_count": post.get("share_count", 0),
                "message": post.get("message", ""),
                "permalink": post.get("permalink", ""),
                "timeline_visibility": post.get("timeline_visibility", ""),
                "commentaires": commentaire_p,
                "sous_commentaires": commentaire_p_sous_commentaire,
            })

        df_fb_metrics = pd.DataFrame(records)
        df_fb_metrics["commentaires_count"] = df_fb_metrics["commentaires"].apply(lambda x: len([m for m in x if m.strip() != ""]))
        df_fb_metrics["sous_commentaires_count"] = df_fb_metrics["sous_commentaires"].apply(
            lambda x: len([y.get("sous_commentaires") for y in x if isinstance(x, list)])
        )
        df_fb_metrics["date"] = pd.to_datetime(df_fb_metrics["created_time"]).dt.strftime("%Y.%m.%d")
        df_fb_metrics["hour"] = pd.to_datetime(df_fb_metrics["created_time"]).dt.strftime("%H:%M:%S")
        df_export = df_fb_metrics.copy()
        df_export.drop(columns=["created_time", "commentaires", "sous_commentaires", "message"], inplace=True)
        df_export.to_excel(ORGANIC_FB_EXCEL_DIR / "fb_post_metrics.xlsx", index=False)
    else:
        print(f"File not found: {filename}")


def concat_excel_post_metrics_and_insights(folder=ORGANIC_FB_EXCEL_DIR):
    filename_insights = "fb_post_insights.xlsx"
    file_name_metrics = "fb_post_metrics.xlsx"

    path_file_insights = folder / filename_insights
    if not path_file_insights.exists():
        print(f"File not found: {path_file_insights}")
        return

    path_file_metrics = folder / file_name_metrics
    if not path_file_metrics.exists():
        print(f"File not found: {path_file_metrics}")
        return

    df_post_insights = pd.read_excel(path_file_insights)
    df_post_metrics = pd.read_excel(path_file_metrics)

    df_concat_post_metrics = pd.merge(df_post_insights, df_post_metrics, on=["post_id", "date", "hour"])
    df_concat_post_metrics["date"] = pd.to_datetime(df_concat_post_metrics["date"]).dt.date.astype(str)
    file_name_export_concat = "fb_merge_insights_metrics.xlsx"
    df_concat_post_metrics.to_excel(folder / file_name_export_concat, index=False)


def main_fb():
    convert_page_json_insight_to_excel()
    convert_post_metrics_json_to_excel()
    convert_post_insight_json_to_excel()
    concat_excel_post_metrics_and_insights()

    send_facebook_to_google_sheet(google_sheet_list=["Post Insights", "Page Insights"])


# IG

def convert_insta_page_insights_to_excel(folder=ORGANIC_IG_JSON_DIR) -> bool:
    ORGANIC_IG_EXCEL_DIR.mkdir(parents=True, exist_ok=True)

    file_name = "ig_page_insights.json"
    file_path = folder / file_name
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"{file_name} is empty")
        return False

    # Filter out rows where value is a dict (demographic breakdowns)
    # and keep only numeric/scalar values for the pivot table
    filtered_data = []
    for row in data:
        value = row.get("value")
        if isinstance(value, dict):
            # Skip demographic breakdowns for now (they need special handling)
            continue
        filtered_data.append(row)

    if not filtered_data:
        print("No numeric Instagram page insights to export")
        return False

    df = pd.DataFrame(filtered_data)
    if df.empty:
        print("No Instagram page insights to export")
        return False

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df_pivot = df.pivot_table(index="date", columns="metric", values="value", aggfunc="max")
    df_pivot.reset_index(inplace=True)
    df_pivot["date"] = pd.to_datetime(df_pivot["date"]).dt.strftime("%Y.%m.%d")

    export_path = ORGANIC_IG_EXCEL_DIR / "ig_page_insights.xlsx"
    df_pivot.to_excel(export_path, index=False)
    return True


def convert_insta_insight_per_media_to_excel(folder=ORGANIC_IG_JSON_DIR):
    file_name = "insights_per_media.json"
    file_path = folder / file_name
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        datas_insight_insta = json.load(f)

    df_insta = pd.DataFrame(datas_insight_insta)
    if df_insta.empty:
        print("No Instagram media insights to export")
        return
    df_insta["timestamp"] = pd.to_datetime(df_insta["timestamp"], errors="coerce")
    df_insta["views"] = pd.to_numeric(df_insta["views"], errors="coerce").fillna(0).astype(int)
    df_insta["date"] = df_insta["timestamp"].dt.date
    df_insta["date"] = pd.to_datetime(df_insta["date"]).dt.strftime("%Y.%m.%d")
    df_insta["hour"] = pd.to_datetime(df_insta["timestamp"]).dt.strftime("%H:%M:%S")
    df_insta["datetime"] = df_insta["date"] + " " + df_insta["hour"]
    df_insta.drop(columns="timestamp", inplace=True)

    df_insta["media_id"] = df_insta["media_id"].astype("string").apply(lambda x: f"'{x}")
    df_insta["reach"] = pd.to_numeric(df_insta["reach"], errors="coerce").fillna(0).astype(int)
    df_insta["profile_visits"] = pd.to_numeric(df_insta["profile_visits"], errors="coerce").fillna(0).astype(int)
    df_insta["follows"] = pd.to_numeric(df_insta["follows"], errors="coerce").fillna(0).astype(int)

    df_stories = extract_stories_excel()
    df_insta = pd.concat([df_insta, df_stories], ignore_index=True)

    df_insta = df_insta.drop(
        columns=[
            "text_comment",
            "time_comment",
            "id_comment",
            "like_count_comment",
            "replies_text_comment",
            "replies_like_count_comment",
            "replies_user_comment",
        ]
    )

    text_fill = {
        "permalink": "-",
        "description_assets": "-",
    }
    numeric_fill = {
        "clics stickers": 0,
        "navigation": 0,
        "clics": 0,
        "saved": 0,
        "link clics": 0,
        "Clics sur un lien": 0,
    }

    for col, value in text_fill.items():
        if col in df_insta.columns:
            df_insta[col] = df_insta[col].fillna(value)

    for col, value in numeric_fill.items():
        if col in df_insta.columns:
            df_insta[col] = pd.to_numeric(df_insta[col], errors="coerce").fillna(value)

    df_insta[["profile_visits", "follows", "clics stickers", "link clics"]] = df_insta[["profile_visits", "follows", "clics stickers", "link clics"]].astype(float)

    export_path = ORGANIC_IG_EXCEL_DIR / "insight_per_media_insta.xlsx"

    df_insta.to_excel(export_path, index=False)


def main_insta():
    page_insights_exported = convert_insta_page_insights_to_excel()
    convert_insta_insight_per_media_to_excel()
    send_insta_post_to_google_sheet()
    if page_insights_exported:
        send_insta_page_insights_to_google_sheet()


if __name__ == "__main__":
    main_fb()
    main_insta()
