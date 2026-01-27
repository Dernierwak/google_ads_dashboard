import json
import time
from pathlib import Path

from core.config import GRAPH_API_VERSION_ORGANIC, ID_CONT_FB, ID_CONT_IG, TOKEN_LONG, TOKEN_LONG_POST
from core.http import fetch_paginated, session_with_retry
from core.logging import get_logger
from core.paths import ORGANIC_IG_JSON_DIR, ORGANIC_FB_JSON_DIR

logger = get_logger(__name__)

for directory in [ORGANIC_IG_JSON_DIR, ORGANIC_FB_JSON_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def get_all_ig_post_id(session=None, ig_id: str | None = None, access_token: str | None = None):
    session = session or session_with_retry()
    ig_id = _require(ig_id or ID_CONT_IG, "ID_CONT_IG")
    access_token = _require(access_token or TOKEN_LONG, "TOKEN_LONG")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{ig_id}/media"
    params = {
        "access_token": access_token,
        "limit": 50,
        "fields": "id,timestamp",
        "pretty": 0,
    }

    items: list[dict] = []
    for data, _ in fetch_paginated(url, params, session):
        items.extend(data.get("data", []))

    payload = {
        "ids": [item.get("id") for item in items],
        "timestamp": [item.get("timestamp") for item in items],
    }

    output = ORGANIC_IG_JSON_DIR / "all_ids_insta.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Finished Instagram posts: %s items", len(items))
    return payload


def get_all_fb_post_ids(session=None, page_id: str | None = None, access_token: str | None = None, since_date: str = "2025-01-01"):
    session = session or session_with_retry()
    page_id = _require(page_id or ID_CONT_FB, "ID_CONT_FB")
    access_token = _require(access_token or TOKEN_LONG_POST, "TOKEN_LONG_POST")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{page_id}/posts"
    params = {
        "access_token": access_token,
        "fields": "id,created_time,permalink",
        "limit": 100,
        "since": since_date,
    }

    posts: list[dict] = []
    for data, _ in fetch_paginated(url, params, session):
        posts.extend(data.get("data", []))

    output = ORGANIC_FB_JSON_DIR / "all_ids_post_fb.json"
    output.write_text(json.dumps(posts, indent=2), encoding="utf-8")
    logger.info("Finished Facebook posts: %s items", len(posts))
    return posts


def get_all_ig_fb_stories_id(session=None, page_id: str | None = None, access_token: str | None = None):
    session = session or session_with_retry()
    page_id = _require(page_id or ID_CONT_FB, "ID_CONT_FB")
    access_token = _require(access_token or TOKEN_LONG_POST, "TOKEN_LONG_POST")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION_ORGANIC}/{page_id}/stories"
    params = {
        "access_token": access_token,
        "limit": 25,
        "fields": "id,media_type,media_url,timestamp,post_id,media_id",
    }

    media_id: list[str] = []
    media_type: list[str] = []
    media_url: list[str] = []
    timestamps: list[str] = []

    for data, headers in fetch_paginated(url, params, session):
        for story in data.get("data", []):
            media_id.append(story.get("id"))
            media_type.append(story.get("media_type"))
            media_url.append(story.get("media_url"))
            timestamps.append(story.get("timestamp"))

        raw_usage = headers.get("x-app-usage") if headers else None
        if raw_usage:
            try:
                usage_data = json.loads(raw_usage)
                total_time = usage_data.get("total_time")
                if total_time and total_time >= 85:
                    logger.warning("Quota close to limit (%s/100). Pausing 1h.", total_time)
                    time.sleep(3600)
            except json.JSONDecodeError:
                pass

    all_media = {
        "story_id": media_id,
        "date": timestamps,
        "media_url": media_url,
        "media_type": media_type,
    }

    output = ORGANIC_IG_JSON_DIR / "id_stories.json"
    output.write_text(json.dumps(all_media, indent=2), encoding="utf-8")
    logger.info("Finished stories: %s items", len(media_id))
    return all_media


if __name__ == "__main__":
    session = session_with_retry()
    get_all_ig_post_id(session)
