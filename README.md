# Meta API Pipeline (Ads + Organic)

This project collects Meta Ads and Organic data, exports to Excel, and sends to Google Sheets.

## Setup

```bash
pip install -r requirements.txt
```

Environment variables are loaded from `.env` (and `.env_open_ai` for the AI agent).

## Structure (V2)

- `src/` Python modules (core, meta ads/organic, dashboards, AI agent)
- `data/` generated JSON/Excel artifacts
- `scripts/` entry points for running pipelines
- `connection/` OAuth helpers

## Run pipelines

Ads:
```bash
python scripts/run_ads.py
python scripts/run_export_ads.py
python scripts/run_sheets_ads.py
```

Organic:
```bash
python scripts/run_organic_fb.py
python scripts/run_organic_ig.py
python scripts/run_export_organic.py
python scripts/run_sheets_organic.py
```

## Dashboards (Streamlit)

```bash
streamlit run src/dashboards/meta_ads.py
streamlit run src/dashboards/google_ads.py
streamlit run src/dashboards/ig_posts.py
streamlit run src/dashboards/ig_comments.py
```
