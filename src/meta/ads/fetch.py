import json
import os
from datetime import datetime
from pathlib import Path

from core.http import session_with_retry

from core.config import GRAPH_API_VERSION_ADS, TOKEN_LONG, TOKEN_LONG_POST, TOKEN_POST
from core.paths import ADS_DIR, ADS_JSON_DIR, ADS_EXCEL_DIR
from .export import transform_json_fb_ads_insight_to_excel
from .sheets import send_ads_facebook_to_google_sheet

URL_TARGET = f"https://graph.facebook.com/{GRAPH_API_VERSION_ADS}/"
TODAY = datetime.today().strftime("%Y-%m-%d")

TOKEN_ADS = TOKEN_LONG
TOKEN_ASSETS = TOKEN_LONG_POST or TOKEN_POST

SESSION = session_with_retry()

for directory in [ADS_DIR, ADS_JSON_DIR, ADS_EXCEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FILE_CAMPAIGNS_ADSETS_ADS = "all_campaigns_adsets_ads.json"


def _require_token(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def get_account_id() -> list[dict]:
    token = _require_token(TOKEN_ADS, "TOKEN_LONG")
    response = SESSION.get(f"{URL_TARGET}me/adaccounts", params={"access_token": token})
    ids_facebook = []

    if response.status_code == 200:
        for account in response.json().get("data", []):
            ids_facebook.append({"account_id": account.get("account_id")})
    else:
        print(f"Error API: {response.status_code} {response.text}")

    print(f"Account ids: {ids_facebook}")
    return ids_facebook


def get_ads_adset_campaigns() -> None:
    token = _require_token(TOKEN_ADS, "TOKEN_LONG")
    fb_account_id = get_account_id()[0].get("account_id")

    url_campaigns = f"{URL_TARGET}act_{fb_account_id}/campaigns"
    url_adset = f"{URL_TARGET}act_{fb_account_id}/adsets"
    url_ads = f"{URL_TARGET}act_{fb_account_id}/ads"

    list_params = {
        "campaigns": {
            "url": url_campaigns,
            "params": {
                "fields": "name,id,created_time,stop_time,start_time,status",
                "access_token": token,
            },
        },
        "adsets": {
            "url": url_adset,
            "params": {
                "fields": "name,id,created_time,start_time,end_time,status,budget_remaining,lifetime_budget,daily_budget",
                "access_token": token,
            },
        },
        "ads": {
            "url": url_ads,
            "params": {
                "fields": "name,id,status,created_time,ad_schedule_end_time,ad_schedule_start_time,campaign_id,campaign_name,adset_id,adset_name,preview_shareable_link",
                "access_token": token,
            },
        },
    }

    all_data_campaigns_adsets_ads: dict[str, list[dict]] = {}

    for key, config in list_params.items():
        all_data: list[dict] = []
        response = SESSION.get(config["url"], params=config["params"])
        while response.status_code == 200:
            data = response.json()
            all_data.extend(data.get("data", []))
            next_url = data.get("paging", {}).get("next")
            if not next_url:
                break
            response = SESSION.get(next_url)

        if all_data:
            all_data_campaigns_adsets_ads[key] = all_data
            print(f"{key} data received")
        else:
            print(f"Error fetching {key}: {response.status_code} - {response.text}")

    with open(ADS_JSON_DIR / FILE_CAMPAIGNS_ADSETS_ADS, "w", encoding="utf-8") as f:
        json.dump(all_data_campaigns_adsets_ads, f)


def clean_json_ads_adset_campaigns() -> None:
    with open(ADS_JSON_DIR / FILE_CAMPAIGNS_ADSETS_ADS, "r", encoding="utf-8") as f:
        all_datas = json.load(f)

    campaigns_data = all_datas.get("campaigns", [])
    clean_campaigns_data = []
    for data in campaigns_data:
        clean_campaigns_data.append({
            "campaigns_name": data.get("name", ""),
            "campaigns_id": data.get("id", ""),
            "created_time": data.get("created_time", ""),
            "start_time": data.get("start_time", ""),
            "stop_time": data.get("stop_time", ""),
            "status": data.get("status", ""),
        })

    with open(ADS_JSON_DIR / "campaigns_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_campaigns_data, f)

    adsets_data = all_datas.get("adsets", [])
    clean_adsets_data = []
    for data in adsets_data:
        clean_adsets_data.append({
            "adset_name": data.get("name", ""),
            "adsets_id": data.get("id", ""),
            "created_time": data.get("created_time", ""),
            "start_time": data.get("start_time", ""),
            "end_time": data.get("end_time", ""),
            "status": data.get("status", ""),
            "budget_remaining": data.get("budget_remaining", ""),
            "lifetime_budget": data.get("lifetime_budget", ""),
            "daily_budget": data.get("daily_budget", ""),
        })

    with open(ADS_JSON_DIR / "adsets_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_adsets_data, f)

    ads_data = all_datas.get("ads", [])
    clean_ads_data = []
    campaigns_id_to_name = {c["campaigns_id"]: c["campaigns_name"] for c in clean_campaigns_data}
    adset_id_to_name = {c["adsets_id"]: c["adset_name"] for c in clean_adsets_data}

    for data in ads_data:
        campaign_id = data.get("campaign_id", "")
        adset_id = data.get("adset_id", "")

        clean_ads_data.append({
            "ads_name": data.get("name", ""),
            "ads_id": data.get("id", ""),
            "created_time": data.get("created_time", ""),
            "status": data.get("status", ""),
            "preview_shareable_link": data.get("preview_shareable_link", ""),
            "campagin_name": campaigns_id_to_name.get(campaign_id),
            "campaign_id": campaign_id,
            "adset_name": adset_id_to_name.get(adset_id),
            "adset_id": adset_id,
        })

    with open(ADS_JSON_DIR / "ads_data.json", "w", encoding="utf-8") as f:
        json.dump(clean_ads_data, f)


def get_insights_for_ads() -> None:
    token = _require_token(TOKEN_ADS, "TOKEN_LONG")
    since = "2025-01-01"
    until = TODAY

    with open(ADS_JSON_DIR / "ads_data.json", "r", encoding="utf-8") as f:
        ads_data = json.load(f)

    ads_ids_2025 = [
        {
            "ads_id": ads.get("ads_id"),
            "ads_name": ads.get("ads_name"),
            "adset_name": ads.get("adset_name"),
            "adsets_id": ads.get("adset_id"),
            "campaigns_name": ads.get("campagin_name"),
            "campaigns_id": ads.get("campaign_id"),
            "preview_shareable_link": ads.get("preview_shareable_link", ""),
        }
        for ads in ads_data
    ]

    all_insights: list[dict] = []
    for ad_info in ads_ids_2025:
        ad_id = ad_info.get("ads_id")
        url = f"{URL_TARGET}{ad_id}/insights"
        params = {
            "access_token": token,
            "fields": "impressions,clicks,spend,ctr,cpc,cpm,reach,actions,date_start,date_stop,inline_link_clicks,inline_link_click_ctr",
            "time_increment": 1,
            "time_range": json.dumps({"since": since, "until": until}),
        }

        response = SESSION.get(url, params=params)
        while response.status_code == 200:
            data = response.json()
            print(f"Insights for ad {ad_id}")

            for insight in data.get("data", []):
                actions = insight.get("actions", [])
                all_insights.append({
                    **ad_info,
                    "ads_id": ad_id,
                    "adsets_id": ad_info.get("adsets_id"),
                    "campaigns_id": ad_info.get("campaigns_id"),
                    "date_start": insight.get("date_start"),
                    "date_stop": insight.get("date_stop"),
                    "impressions": insight.get("impressions"),
                    "clicks": insight.get("clicks"),
                    "inline_link_clicks": insight.get("inline_link_clicks", "0"),
                    "spend": insight.get("spend"),
                    "ctr": insight.get("ctr"),
                    "cpc": insight.get("cpc"),
                    "cpm": insight.get("cpm"),
                    "reach": insight.get("reach"),
                    "page_engagement": next((a.get("value") for a in actions if a.get("action_type") == "page_engagement"), ""),
                    "link_click": next((a.get("value") for a in actions if a.get("action_type") == "link_click"), ""),
                    "post_engagement": next((a.get("value") for a in actions if a.get("action_type") == "post_engagement"), ""),
                    "omni_landing_page_view": next((a.get("value") for a in actions if "omni_landing_page_view" in a.get("action_type", "")), ""),
                    "post_reaction": next((a.get("value") for a in actions if "post_reaction" in a.get("action_type", "")), ""),
                })

            next_url = data.get("paging", {}).get("next")
            if next_url:
                response = SESSION.get(next_url)
            else:
                break

    with open(ADS_JSON_DIR / "ads_insights.json", "w", encoding="utf-8") as f:
        json.dump(all_insights, f, indent=2)


def get_assets_for_ads() -> None:
    token = _require_token(TOKEN_ASSETS, "TOKEN_LONG_POST")
    since = "2025-01-01"
    until = TODAY

    with open(ADS_JSON_DIR / "ads_data.json", "r", encoding="utf-8") as f:
        ads_data = json.load(f)

    ads_ids_2025 = [
        {
            "ads_id": ads.get("ads_id"),
            "ads_name": ads.get("ads_name"),
            "adsets_id": ads.get("adset_id"),
            "campaigns_id": ads.get("campaign_id"),
            "preview_shareable_link": ads.get("preview_shareable_link"),
        }
        for ads in ads_data
    ]

    all_insights: list[dict] = []
    for ad_info in ads_ids_2025:
        ad_id = ad_info.get("ads_id")
        url = f"{URL_TARGET}{ad_id}/adcreatives"
        params = {
            "access_token": token,
            "fields": "id,title,body,image_url,image_hash,call_to_action,name,object_url,link_url,object_type,object_story_spec{link_data,page_id},asset_feed_spec,effective_object_story_id,instagram_permalink_url",
            "time_increment": 1,
            "time_range": json.dumps({"since": since, "until": until}),
        }

        response = SESSION.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            assets = data.get("data", [])
            for asset in assets:
                all_insights.append({
                    **ad_info,
                    "id_assets": asset.get("id", ""),
                    "name": asset.get("name", ""),
                    "body": asset.get("body", ""),
                    "title": asset.get("title", ""),
                    "title_assets": [x.get("text", "") for x in asset.get("asset_feed_spec", {}).get("titles", [])],
                    "body_assets": [x.get("text", "") for x in asset.get("asset_feed_spec", {}).get("bodies", [])],
                    "image_url": asset.get("image_url", ""),
                    "call_to_action": asset.get("call_to_action", {}).get("type", ""),
                    "object_type": asset.get("object_type", ""),
                    "effective_object_story_id": asset.get("effective_object_story_id", ""),
                    "instagram_permalink_url": asset.get("instagram_permalink_url", ""),
                })

    with open(ADS_JSON_DIR / "ads_assets.json", "w", encoding="utf-8") as f:
        json.dump(all_insights, f, indent=2)


def get_all_comments(story_id: str) -> list[str]:
    token = _require_token(TOKEN_ASSETS, "TOKEN_LONG_POST")
    comments_all: list[str] = []
    url = f"{URL_TARGET}{story_id}/comments"
    params = {
        "access_token": token,
        "summary": "true",
        "limit": 10,
    }

    while url:
        response = SESSION.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            comments = data.get("data", [])
            comments_all.extend([c.get("message", "") for c in comments])
            url = data.get("paging", {}).get("next", None)
            params = None
        else:
            print(f"Pagination error for {story_id} - {response.status_code}")
            break

    return comments_all


def get_interaction_ads() -> None:
    token = _require_token(TOKEN_ASSETS, "TOKEN_LONG_POST")
    with open(ADS_JSON_DIR / "ads_assets.json", "r", encoding="utf-8") as f:
        ads_assets = json.load(f)
    with open(ADS_JSON_DIR / "campaigns_data.json", "r", encoding="utf-8") as f:
        campaigns_data = json.load(f)
    with open(ADS_JSON_DIR / "adsets_data.json", "r", encoding="utf-8") as f:
        adsets_data = json.load(f)

    campaign_id_to_name = {c["campaigns_id"]: c["campaigns_name"] for c in campaigns_data}
    adset_id_to_name = {a["adsets_id"]: a["adset_name"] for a in adsets_data}

    all_interactions: list[dict] = []

    for asset in ads_assets:
        story_id = asset.get("effective_object_story_id")
        if not story_id:
            continue

        campaign_name = campaign_id_to_name.get(asset.get("campaigns_id"), "")
        adset_name = adset_id_to_name.get(asset.get("adsets_id"), "")

        url = f"{URL_TARGET}{story_id}"
        params = {
            "access_token": token,
            "fields": "comments.summary(true),reactions.summary(true),shares,dynamic_posts",
        }

        response = SESSION.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_interactions.append({
                "ads_id": asset.get("ads_id"),
                "ads_name": asset.get("ads_name"),
                "adsets_id": asset.get("adsets_id"),
                "adsets_name": adset_name,
                "campaigns_id": asset.get("campaigns_id"),
                "campaigns_name": campaign_name,
                "story_id": story_id,
                "comments_count": data.get("comments", {}).get("summary", {}).get("total_count", 0),
                "reactions_count": data.get("reactions", {}).get("summary", {}).get("total_count", 0),
                "shares_count": data.get("shares", {}).get("count", 0),
                "preview_shareable_link": asset.get("preview_shareable_link", ""),
                "comments_messages": get_all_comments(story_id),
            })
        else:
            print(f"Error for story ID {story_id} - {response.status_code}")

    with open(ADS_JSON_DIR / "ads_interactions.json", "w", encoding="utf-8") as f:
        json.dump(all_interactions, f, indent=2)


def main() -> None:
    print("Fetching ads with campaign/adset context")
    get_ads_adset_campaigns()
    print("Cleaning data and exporting JSON")
    clean_json_ads_adset_campaigns()
    print("Fetching ads insights")
    get_insights_for_ads()
    transform_json_fb_ads_insight_to_excel()
    send_ads_facebook_to_google_sheet()
    print("Done")


def try_news_adding() -> None:
    get_assets_for_ads()
    get_interaction_ads()


if __name__ == "__main__":
    # main()
    try_news_adding()
