# server.py
from flask import Flask, request
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
APP_ID = os.getenv("ID_APP")
APP_SECRET = os.getenv("SECRET_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")
URL_TOKEN = os.getenv("URL_TOKEN")
ENV_PATH = ".env"

app = Flask(__name__)

def update_env_with_token(token):
    """Update .env file with the new TOKEN."""
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as file:
            lines = file.readlines()

    token_updated = False
    for i, line in enumerate(lines):
        if line.startswith("TOKEN="):
            lines[i] = f"TOKEN={token}\n"
            token_updated = True

    if not token_updated:
        lines.append(f"TOKEN={token}\n")

    with open(ENV_PATH, "w") as file:
        file.writelines(lines)

    print("✅ Access token saved to .env file.")

@app.route("/")
def callback():
    code = request.args.get("code")
    state = request.args.get("state")

    if not code:
        return "❌ No code received."

    # Exchange code for access_token
    params = {
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code
    }

    print("🔄 Exchanging code for access token...")

    response = requests.get(URL_TOKEN, params=params)

    if response.status_code == 200:
        access_token = response.json().get("access_token")
        if access_token:
            update_env_with_token(access_token)
            return f"✅ Authorization successful! Access token saved.\n\nToken: {access_token}"
        else:
            return "❌ No access token found in response."
    else:
        print(response.text)
        return f"❌ Error exchanging code: {response.status_code}"

if __name__ == "__main__":
    app.run(port=5000)
