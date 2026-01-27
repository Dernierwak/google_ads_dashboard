import os
from dotenv import load_dotenv
from .paths import ROOT

load_dotenv(ROOT / ".env")

GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v22.0")
GRAPH_API_VERSION_ADS = os.getenv("GRAPH_API_VERSION_ADS", "v22.0")
GRAPH_API_VERSION_ORGANIC = os.getenv("GRAPH_API_VERSION_ORGANIC", "v22.0")
GRAPH_API_VERSION_IG_PAGE = os.getenv("GRAPH_API_VERSION_IG_PAGE", "v23.0")
GRAPH_API_VERSION_AUTH = os.getenv("GRAPH_API_VERSION_AUTH", "v24.0")


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


TOKEN_LONG = get_env("TOKEN_LONG")
TOKEN_LONG_POST = get_env("TOKEN_LONG_POST")
TOKEN_POST = get_env("TOKEN_POST")
TOKEN = get_env("TOKEN")

ID_CONT_FB = get_env("ID_CONT_FB")
ID_CONT_IG = get_env("ID_CONT_IG")

ID_APP = get_env("ID_APP")
SECRET_KEY = get_env("SECRET_KEY")
REDIRECT_URI = get_env("REDIRECT_URI")
URL_TOKEN = get_env("URL_TOKEN")

