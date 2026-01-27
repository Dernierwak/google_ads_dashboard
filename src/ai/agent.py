from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

import pandas as pd

# Load .env_open_ai only
ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env_open_ai"
load_dotenv(ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PREFIX = """
You are a senior digital marketing expert specialized in Facebook Ads.

Rules:
- Use only the dataframe data.
- Provide numeric insights with exact column names.
- Provide prioritized, actionable recommendations.
- Avoid generic advice.

Response format:
1. Executive summary (max 5 lines)
2. Key insights (numbered list with precise numbers)
3. Action plan (step-by-step, prioritized)
"""


def invok_agent(df: pd.DataFrame):
    return create_pandas_dataframe_agent(
        llm=ChatOpenAI(model="gpt-5-mini"),
        prefix=PREFIX,
        df=df,
        allow_dangerous_code=True,
        verbose=True,
    )
