import requests
from requests.adapters import HTTPAdapter, Retry


def session_with_retry(total: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    retry = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "TRACE"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_paginated(url: str, params: dict | None, session: requests.Session):
    next_url = url
    next_params = params
    while next_url:
        response = session.get(next_url, params=next_params)
        if not response.ok:
            raise RuntimeError(f"GET {next_url} -> {response.status_code}: {response.text}")
        data = response.json()
        yield data, response.headers
        next_url = data.get("paging", {}).get("next")
        next_params = None

