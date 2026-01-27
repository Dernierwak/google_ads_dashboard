import os
import sys
import logging

import requests
from dotenv import load_dotenv, set_key


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


GRAPH_API_VERSION = "v24.0"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def exchange_for_long_lived(short_lived_token: str, client_id: str, client_secret: str) -> str:
    """Exchange a short-lived token for a long-lived one; raises on error."""
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "fb_exchange_token": short_lived_token,
    }
    resp = requests.get(url, params=params)
    if not resp.ok:
        raise RuntimeError(f"Exchange failed {resp.status_code}: {resp.text}")
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {data}")
    return token


def main():
    load_dotenv()
    client_id = require_env("ID_APP")
    client_secret = require_env("SECRET_KEY")

    short_user = require_env("TOKEN")
    short_page = require_env("TOKEN_POST")

    long_user = exchange_for_long_lived(short_user, client_id, client_secret)
    set_key(dotenv_path=".env", key_to_set="TOKEN_LONG", value_to_set=long_user)
    log.info("Updated TOKEN_LONG in .env")

    long_page = exchange_for_long_lived(short_page, client_id, client_secret)
    set_key(dotenv_path=".env", key_to_set="TOKEN_LONG_POST", value_to_set=long_page)
    log.info("Updated TOKEN_LONG_POST in .env")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("Token exchange failed: %s", exc)
        sys.exit(1)
