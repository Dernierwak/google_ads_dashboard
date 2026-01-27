import pandas as pd

from core.paths import ADS_DIR, ADS_JSON_DIR, ADS_EXCEL_DIR

for directory in [ADS_DIR, ADS_JSON_DIR, ADS_EXCEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def _read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_json(path)


def transform_json_fb_ads_insight_to_excel() -> None:
    df_fb_ads_insights = _read_json(ADS_JSON_DIR / "ads_insights.json")
    df_campaigns = _read_json(ADS_JSON_DIR / "campaigns_data.json")
    df_adsets = _read_json(ADS_JSON_DIR / "adsets_data.json")

    assets_path = ADS_JSON_DIR / "ads_assets.json"
    if assets_path.exists():
        df_assets = pd.read_json(assets_path)
    else:
        df_assets = pd.DataFrame(columns=["campaigns_id", "name", "body"])

    df_merge = pd.merge(
        df_fb_ads_insights,
        df_adsets[["adsets_id", "adset_name"]],
        on="adsets_id",
        how="left",
    )

    df_merge = pd.merge(
        df_merge,
        df_campaigns[["campaigns_id", "campaigns_name"]],
        on="campaigns_id",
        how="left",
    )

    if not df_assets.empty:
        df_merge = pd.merge(
            df_merge,
            df_assets[["campaigns_id", "name", "body"]],
            on="campaigns_id",
            how="left",
        )

    df_merge.fillna("0", inplace=True)
    df_merge["campaigns_id"] = df_merge["campaigns_id"].astype("string")
    df_merge["adsets_id"] = df_merge["adsets_id"].astype("string")
    df_merge["ads_id"] = df_merge["ads_id"].astype("string")
    df_merge["ctr"] = df_merge["ctr"].astype(float).round(2)
    df_merge["cpm"] = df_merge["cpm"].astype(float).round(2)
    df_merge["cpc"] = df_merge["cpc"].astype(float).round(2)
    df_merge["spend"] = df_merge["spend"].astype(float)
    df_merge["page_engagement"] = df_merge["page_engagement"].str.replace("", "0").astype(int)
    df_merge["post_engagement"] = df_merge["post_engagement"].str.replace("", "0").astype(int)
    df_merge["link_click"] = df_merge["link_click"].str.replace("", "0").astype(int)
    df_merge["omni_landing_page_view"] = df_merge["omni_landing_page_view"].str.replace("", "0").astype(int)
    df_merge["post_reaction"] = df_merge["post_reaction"].str.replace("", "0").astype(int)
    df_merge["campaigns_id"] = df_merge["campaigns_id"].apply(lambda x: f"'{x}")
    df_merge["adsets_id"] = df_merge["adsets_id"].apply(lambda x: f"'{x}")
    df_merge["ads_id"] = df_merge["ads_id"].apply(lambda x: f"'{x}")
    df_merge["date_start"] = df_merge["date_start"].str.replace("-", ".")
    df_merge["date_stop"] = df_merge["date_stop"].str.replace("-", ".")
    df_merge = df_merge.drop(columns=["adset_name_y", "campaigns_name_y"])
    df_merge.rename(
        columns={
            "adset_name_x": "adset_name",
            "campaigns_name_x": "campaigns_name",
        },
        inplace=True,
    )

    df_merge.to_excel(ADS_EXCEL_DIR / "final_ads_insights.xlsx", index=False)


if __name__ == "__main__":
    transform_json_fb_ads_insight_to_excel()
