import json
from datetime import datetime, timedelta
from pathlib import Path


from core.config import GRAPH_API_VERSION_ORGANIC, TOKEN_LONG_POST, ID_CONT_FB
from core.paths import ORGANIC_FB_JSON_DIR
from core.http import session_with_retry
from .common import get_all_fb_post_ids
from .export import main_fb

PAGE_TOKEN = TOKEN_LONG_POST
ID_CONT_FB = ID_CONT_FB
session = session_with_retry()


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def get_page_insights_chunked():
    start_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_date = datetime.today()
    delta = timedelta(days=93)
    all_data = []

    while start_date < end_date:
        chunk_end = min(start_date + delta, end_date)
        since = start_date.strftime("%Y-%m-%d")
        until = chunk_end.strftime("%Y-%m-%d")

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{_require(ID_CONT_FB, 'ID_CONT_FB')}/insights"
        params = {
            "metric": (
                "page_impressions_unique,"
                "page_views_total,page_posts_impressions_organic,page_daily_follows,"
                "page_fan_adds_by_paid_non_paid_unique,page_follows,page_daily_unfollows_unique"
            ),
            "period": "day",
            "since": since,
            "until": until,
            "access_token": _require(PAGE_TOKEN, "TOKEN_LONG_POST"),
        }

        response = session.get(url, params=params)
        if response.status_code == 200:
            all_data.extend(response.json().get("data", []))
        else:
            print(f"Error {response.status_code}: {response.text}")

        start_date = chunk_end + timedelta(days=1)

    ORGANIC_FB_JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(ORGANIC_FB_JSON_DIR / "fb_page_insights.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    return all_data


def get_post_metrics():
    with open(ORGANIC_FB_JSON_DIR / "all_ids_post_fb.json", "r", encoding="utf-8") as f:
        posts = json.load(f)

    posts_2025 = [p for p in posts if p.get("created_time", "").startswith("2025")]
    post_ids = [p["id"] for p in posts_2025]

    metrics = []
    for post_id in post_ids:
        url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{post_id}"
        params = {
            "fields": (
                "shares,created_time,id,message,permalink_url,"
                "timeline_visibility,subscribed,comments{id,message,comment_count,comments}"
            ),
            "access_token": _require(PAGE_TOKEN, "TOKEN_LONG_POST"),
        }
        response = session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            metrics.append({
                "post_id": post_id,
                "created_time": data.get("created_time"),
                "share_count": data.get("shares", {}).get("count", 0),
                "message": data.get("message", ""),
                "permalink": data.get("permalink_url", ""),
                "timeline_visibility": data.get("timeline_visibility", ""),
                "subscribed": data.get("subscribed", ""),
                "comments_data": data.get("comments", []),
            })
        else:
            print(f"Error {response.status_code} Metrics for {post_id}: {response.text}")

    with open(ORGANIC_FB_JSON_DIR / "fb_posts_metrics_2025.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def get_post_insights():
    with open(ORGANIC_FB_JSON_DIR / "all_ids_post_fb.json", "r", encoding="utf-8") as f:
        posts = json.load(f)

    posts_2025 = [p for p in posts if p.get("created_time", "").startswith("2025")]

    insights_all = []
    for post in posts_2025:
        post_id = post["id"]
        created_time = post.get("created_time")

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{post_id}/insights"
        params = {
            "metric": (
                "post_impressions_unique,post_reactions_like_total,"
                "post_reactions_by_type_total,post_clicks"
            ),
            "access_token": _require(PAGE_TOKEN, "TOKEN_LONG_POST"),
        }
        response = session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            insights_all.append({
                "post_id": post_id,
                "created_time": created_time,
                "insights": data.get("data", []),
            })
        else:
            print(f"Error {response.status_code} for post {post_id}: {response.text}")

    with open(ORGANIC_FB_JSON_DIR / "fb_posts_insights_2025.json", "w", encoding="utf-8") as f:
        json.dump(insights_all, f, indent=2)

    return insights_all


def get_stories_inst_fb_insights():
    dir_json = ORGANIC_FB_JSON_DIR.parent / "stories"
    dir_json.mkdir(parents=True, exist_ok=True)
    file_name = "id_stories.json"

    with open(dir_json / file_name, "r", encoding="utf-8") as f:
        ids_stories_insta_fb = json.load(f)

    id_stories_insta_fb_2025 = {
        sid: date
        for sid, date in zip(ids_stories_insta_fb["stories_id"], ids_stories_insta_fb["date"])
        if date.startswith("2024")
    }

    all_insights_id_insta_fb = []

    for ids_2025 in list(id_stories_insta_fb_2025.keys())[:2]:
        url_target = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{ids_2025}/insights"
        params = {"access_token": _require(PAGE_TOKEN, "TOKEN_LONG_POST")}

        response = session.get(url_target, params=params)

        if response.status_code == 200:
            data = response.json()
            insights = data.get("data", [])
            all_insights_id_insta_fb.append({
                "story_id": ids_2025,
                "created_date": id_stories_insta_fb_2025[ids_2025],
                "page_story_impressions_by_story_id": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "page_story_impressions_by_story_id"
                    for v in d.get("values", [])
                ],
                "pages_fb_story_sticker_interactions": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "pages_fb_story_sticker_interactions"
                    for v in d.get("values", [])
                ],
                "pages_fb_story_replies": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "pages_fb_story_replies"
                    for v in d.get("values", [])
                ],
                "pages_fb_story_thread_lightweight_reactions": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "pages_fb_story_thread_lightweight_reactions"
                    for v in d.get("values", [])
                ],
                "pages_fb_story_shares": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "pages_fb_story_shares"
                    for v in d.get("values", [])
                ],
                "page_story_impressions_by_story_id_unique": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "page_story_impressions_by_story_id_unique"
                    for v in d.get("values", [])
                ],
                "story_interaction": [
                    v.get("value", "")
                    for d in insights
                    if d.get("name") == "story_interaction"
                    for v in d.get("values", [])
                ],
            })
        else:
            print(f"Error for story {ids_2025}: {response.status_code} - {response.text}")

    with open(dir_json / "fb_insta_stories_insights_2025.json", "w", encoding="utf-8") as f:
        json.dump(all_insights_id_insta_fb, f, indent=2)


def main():
    get_all_fb_post_ids(session, page_id=ID_CONT_FB, access_token=PAGE_TOKEN, since_date="2025-01-01")

    get_page_insights_chunked()
    get_post_metrics()
    get_post_insights()

    main_fb()


if __name__ == "__main__":
    main()
