from flask import Flask, request
import os
import requests
from dotenv import load_dotenv

from core.paths import ROOT

load_dotenv(ROOT / ".env")

APP_ID = os.getenv("ID_APP")
APP_SECRET = os.getenv("SECRET_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")
URL_TOKEN = os.getenv("URL_TOKEN")
ENV_PATH = ROOT / ".env"

app = Flask(__name__)


def update_env_with_token(token: str) -> None:
    lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()

    token_updated = False
    for i, line in enumerate(lines):
        if line.startswith("TOKEN="):
            lines[i] = f"TOKEN={token}\n"
            token_updated = True

    if not token_updated:
        lines.append(f"TOKEN={token}\n")

    with open(ENV_PATH, "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Access token saved to .env file")


@app.route("/")
def callback():
    code = request.args.get("code")

    if not code:
        return "No code received"

    params = {
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }

    response = requests.get(URL_TOKEN, params=params)

    if response.status_code == 200:
        access_token = response.json().get("access_token")
        if access_token:
            update_env_with_token(access_token)
            return f"Authorization successful. Token: {access_token}"
        return "No access token found in response"

    return f"Error exchanging code: {response.status_code}"


if __name__ == "__main__":
    app.run(port=5000)
