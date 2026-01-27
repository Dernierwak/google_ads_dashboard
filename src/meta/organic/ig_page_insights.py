import json
from datetime import datetime, timedelta

import requests

from core.config import ID_CONT_IG, TOKEN_LONG_POST, GRAPH_API_VERSION_IG_PAGE
from core.http import session_with_retry
from core.logging import get_logger
from core.paths import ORGANIC_IG_JSON_DIR

log = get_logger(__name__)


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing env var: {name}")
    return value


def fetch_ig_page_insights(ig_id: str, access_token: str, start_date: str, end_date: str | None = None):
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
    start = datetime.strptime(start_date, "%Y-%m-%d")
    delta = timedelta(days=30)

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION_IG_PAGE}/{ig_id}/insights"

    # Metrics that require metric_type=total_value (no period parameter)
    metrics_total = [
        "website_clicks",
        "profile_views",
        "accounts_engaged",
        "total_interactions",
        "likes",
        "comments",
        "shares",
        "saves",
        "replies",
        "follows_and_unfollows",
        "profile_links_taps",
        "views",
        "reposts",
        "content_views",
    ]

    rows: list[dict] = []
    session = session_with_retry()

    while start <= end:
        chunk_end = min(start + delta, end)

        # Reach metric (works with period=day, no 30-day limit)
        params_reach = {
            "metric": "reach",
            "period": "day",
            "since": start.strftime("%Y-%m-%d"),
            "until": chunk_end.strftime("%Y-%m-%d"),
            "access_token": access_token,
        }
        resp_reach = session.get(url, params=params_reach)
        if not resp_reach.ok:
            log.warning("GET %s (reach) -> %s: %s", url, resp_reach.status_code, resp_reach.text)
        else:
            data_reach = resp_reach.json().get("data", [])
            for metric in data_reach:
                name = metric.get("name")
                for val in metric.get("values", []):
                    rows.append({
                        "metric": name,
                        "date": val.get("end_time", "")[:10],
                        "value": val.get("value"),
                    })

        # Total value metrics
        if metrics_total:
            params_tot = {
                "metric": ",".join(metrics_total),
                "period": "day",
                "metric_type": "total_value",
                "since": start.strftime("%Y-%m-%d"),
                "until": chunk_end.strftime("%Y-%m-%d"),
                "access_token": access_token,
            }
            resp_tot = session.get(url, params=params_tot)
            if not resp_tot.ok:
                log.warning("GET %s (total) -> %s: %s", url, resp_tot.status_code, resp_tot.text)
            else:
                data_tot = resp_tot.json().get("data", [])
                for metric in data_tot:
                    name = metric.get("name")
                    total_values = metric.get("total_value", {})
                    if isinstance(total_values, dict):
                        rows.append({
                            "metric": name,
                            "date": chunk_end.strftime("%Y-%m-%d"),
                            "value": total_values,
                        })
                    else:
                        rows.append({
                            "metric": name,
                            "date": chunk_end.strftime("%Y-%m-%d"),
                            "value": total_values,
                        })

        log.info("Fetched %s to %s", start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        start = chunk_end + timedelta(days=1)

    # Follower count - only last 30 days (API limitation)
    follower_until = datetime.today() - timedelta(days=1)
    follower_since = follower_until - timedelta(days=29)
    params_followers = {
        "metric": "follower_count",
        "period": "day",
        "since": follower_since.strftime("%Y-%m-%d"),
        "until": follower_until.strftime("%Y-%m-%d"),
        "access_token": access_token,
    }
    resp_fc = session.get(url, params=params_followers)
    if not resp_fc.ok:
        log.warning("GET %s (follower_count) -> %s: %s", url, resp_fc.status_code, resp_fc.text)
    else:
        data_fc = resp_fc.json().get("data", [])
        for metric in data_fc:
            name = metric.get("name")
            for val in metric.get("values", []):
                rows.append({
                    "metric": name,
                    "date": val.get("end_time", "")[:10],
                    "value": val.get("value"),
                })

    ORGANIC_IG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ORGANIC_IG_JSON_DIR / "ig_page_insights.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        out_csv = ORGANIC_IG_JSON_DIR / "ig_page_insights.csv"
        df.to_csv(out_csv, index=False)
    except ImportError:
        log.warning("pandas not installed: CSV export skipped")

    return rows


def main():
    ig_id = _require(ID_CONT_IG, "ID_CONT_IG")
    token = _require(TOKEN_LONG_POST, "TOKEN_LONG_POST")
    fetch_ig_page_insights(ig_id, token, start_date="2025-01-01")


if __name__ == "__main__":
    main()
