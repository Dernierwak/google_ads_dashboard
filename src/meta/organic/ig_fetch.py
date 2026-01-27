import json
import os
from datetime import datetime

import pandas as pd

from core.config import GRAPH_API_VERSION_ORGANIC, TOKEN_LONG, TOKEN_LONG_POST, ID_CONT_IG
from core.http import session_with_retry
from core.paths import ORGANIC_IG_JSON_DIR
from .common import get_all_ig_post_id
from .export import main_insta
from .ig_page_insights import fetch_ig_page_insights

TOKEN = TOKEN_LONG
URL_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/"

ALL_IDS_PATH = ORGANIC_IG_JSON_DIR / "all_ids_insta.json"
INSIGHTS_MEDIA_PATH = ORGANIC_IG_JSON_DIR / "insights_per_media.json"
DAILY_INSIGHTS_PATH = ORGANIC_IG_JSON_DIR / "insights_daily.json"

session = session_with_retry()


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def parse_media_timestamp(raw_ts):
    if not raw_ts:
        return None
    cleaned = raw_ts.replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    print(f"Unrecognized timestamp format: {raw_ts}")
    return None


def load_media_ids(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{filepath} is empty.")
        all_ids_timestamp = json.loads(content)

    ids = all_ids_timestamp.get("ids", [])
    timestamps = all_ids_timestamp.get("timestamp", [])

    media_ids_2025 = []
    for id_, ts in zip(ids, timestamps):
        parsed_ts = parse_media_timestamp(ts)
        if parsed_ts and parsed_ts.year == 2025:
            media_ids_2025.append(id_)

    return media_ids_2025


def insights_insta_post(medias_id: list):
    media_infos = get_media_type_and_timestamp(medias_id)

    media_infos = [
        media for media in media_infos
        if media.get("parsed_timestamp") and media["parsed_timestamp"].year == 2025
    ]
    media_infos = sorted(media_infos, key=lambda x: x["parsed_timestamp"])

    all_insights = []

    for media in media_infos:
        media_product_type = media["media_product_type"]
        media_id = media["id"]
        timestamp = media["timestamp"]
        media_type = media.get("media_type")
        permalink = media["permalink"]
        metrics = get_valid_metrics_for_type(media_product_type, media_type)

        if not metrics:
            continue

        entry = {
            "media_id": media_id,
            "media_product_type": media_product_type,
            "media_type": media_type,
            "timestamp": timestamp,
            "permalink": permalink,
        }

        params_assets = {"access_token": _require(TOKEN, "TOKEN_LONG"), "fields": "caption"}
        response_assets = session.get(f"{URL_BASE}{media_id}", params=params_assets)
        if response_assets.status_code != 200:
            print(f"Assets error: {response_assets.status_code}")

        post_assets = {
            "description_assets": response_assets.json().get("caption", "")
        }
        entry.update(post_assets)

        params = {"access_token": _require(TOKEN, "TOKEN_LONG"), "metric": metrics}
        url = f"{URL_BASE}{media_id}/insights"
        response = session.get(url=url, params=params)

        if response.status_code == 200:
            data = response.json().get("data", [])
            insights_data = {
                insight["name"]: insight.get("values", [{}])[0].get("value")
                for insight in data if "name" in insight
            }
            entry.update(insights_data)
        else:
            print(f"Error retrieving insights for media {media_id}: {response.status_code} {response.text}")

        comments = []
        next_url = f"{URL_BASE}{media_id}/comments"
        params = {"access_token": _require(TOKEN, "TOKEN_LONG"), "fields": "like_count,text,id,timestamp,replies{like_count,text,id,timestamp}"}
        while next_url:
            response = session.get(url=next_url, params=params)
            if response.status_code != 200:
                print(f"Error retrieving comments for media {media_id}: {response.status_code} {response.text}")
                break
            data = response.json()
            comments.extend(data.get("data", []))
            next_url = data.get("paging", {}).get("next")

        entry.update({
            "text_comment": [c.get("text", "") for c in comments],
            "time_comment": [c.get("timestamp", "") for c in comments],
            "id_comment": [c.get("id", "") for c in comments],
            "like_count_comment": [c.get("like_count", "") for c in comments],
            "replies_text_comment": [
                [x.get("text", "") for x in c.get("replies", {}).get("data", [])]
                for c in comments
            ],
            "replies_like_count_comment": [
                [x.get("like_count", "") for x in c.get("replies", {}).get("data", [])]
                for c in comments
            ],
            "replies_user_comment": [
                [x.get("user", {}).get("id", "") for x in c.get("replies", {}).get("data", [])]
                for c in comments
            ],
        })

        all_insights.append(entry)

    with open(INSIGHTS_MEDIA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_insights, f, indent=2)

    return all_insights


def get_media_type_and_timestamp(medias_id: list):
    medias_info = []
    for media_id in medias_id:
        url = f"{URL_BASE}{media_id}"
        params = {
            "fields": "media_product_type,media_type,timestamp,permalink",
            "access_token": _require(TOKEN, "TOKEN_LONG"),
        }
        response = session.get(url, params=params)
        if response.status_code == 200:
            response_data = response.json()
            parsed_ts = parse_media_timestamp(response_data.get("timestamp"))

            medias_info.append({
                "media_product_type": response_data.get("media_product_type"),
                "timestamp": response_data.get("timestamp"),
                "permalink": response_data.get("permalink"),
                "media_type": response_data.get("media_type"),
                "parsed_timestamp": parsed_ts,
                "id": media_id,
            })
        else:
            print(f"Error retrieving media {media_id}: {response.status_code} {response.text}")
    return medias_info


def get_valid_metrics_for_type(media_product_type, media_type=None):
    if media_product_type == "REELS":
        return "reach,likes,comments,shares,saved,views"
    if media_product_type == "FEED":
        return "reach,saved,likes,comments,shares,profile_visits,follows,views"
    if media_product_type == "VIDEO":
        return "reach,saved,likes,comments,shares,views"
    if media_product_type == "STORY":
        return "impressions,reach,replies,exits,taps_forward,taps_back,follows,views"
    return None


def daily_performance():
    if not os.path.exists(INSIGHTS_MEDIA_PATH):
        print(f"File {INSIGHTS_MEDIA_PATH} not found.")
        return

    with open(INSIGHTS_MEDIA_PATH, "r", encoding="utf-8") as f:
        new_datas = json.load(f)

    if os.path.exists(DAILY_INSIGHTS_PATH):
        with open(DAILY_INSIGHTS_PATH, "r", encoding="utf-8") as f:
            update_datas = json.load(f)
    else:
        update_datas = []

    existing_keys = {(entry["media_id"], entry["date"]) for entry in update_datas}
    new_data_all = [entry for entry in new_datas if (entry["media_id"], entry["date"]) not in existing_keys]

    final_data = update_datas + new_data_all

    with open(DAILY_INSIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"Added {len(new_data_all)} new entries to {DAILY_INSIGHTS_PATH}")


def main():
    print("Step 1: Scraping all Instagram Post IDs")
    get_all_ig_post_id(session=session, ig_id=ID_CONT_IG, access_token=TOKEN)

    print("Step 2: Scraping insights for all posts")
    media_ids_2025 = load_media_ids(ALL_IDS_PATH)
    insights_insta_post(media_ids_2025)

    print("Step 2b: Scraping Instagram page insights")
    fetch_ig_page_insights(
        _require(ID_CONT_IG, "ID_CONT_IG"),
        _require(TOKEN_LONG_POST, "TOKEN_LONG_POST"),
        start_date="2025-01-01",
    )

    print("Step 3: Reformatting timestamps for Excel")
    main_insta()

    print("All operations completed successfully")


if __name__ == "__main__":
    main()
