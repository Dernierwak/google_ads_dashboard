# auth.py
import requests
import os
from dotenv import load_dotenv
import json
import webbrowser
import secrets


load_dotenv()

APP_ID = os.getenv("ID_APP")
APP_SECRET = os.getenv("SECRET_KEY")
URL_TOKEN = os.getenv("URL_TOKEN")

generated_state = secrets.token_urlsafe(16)

def connect_flow():

    url_oauth = "https://www.facebook.com/v22.0/dialog/oauth"
    print(APP_ID)

    params = {
        "client_id": APP_ID,
        "redirect_uri": "https://localhost:500",
        "scope": "pages_show_list,ads_read,ads_management,business_management,instagram_basic,instagram_manage_insights",
        "response_type": "code",
        "state": generated_state
    }

    full_url = url_oauth + "?" + "&".join([f"{key}={value}" for key, value in params.items()])
    print(full_url)
    
    # Open the URL in the default browser
    print(f"Opening browser for OAuth flow...")
    webbrowser.open(full_url)
    

if __name__ == "__main__":
    connect_flow()