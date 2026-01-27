import os
from pathlib import Path

import pandas as pd

from core.paths import ORGANIC_STORIES_CSV_DIR


def extract_stories_excel() -> pd.DataFrame:
    pd.set_option("display.max_columns", None)
    path_dir = ORGANIC_STORIES_CSV_DIR
    path_dir.mkdir(parents=True, exist_ok=True)

    expected_columns = [
        "media_id",
        "media_product_type",
        "views",
        "reach",
        "profile_visits",
        "shares",
        "likes",
        "follows",
        "comments",
        "link clics",
        "date",
        "clics stickers",
        "navigation",
        "hour",
        "datetime",
        "media_type",
    ]

    list_file = os.listdir(path_dir)
    excel_list = [f for f in list_file if f.endswith(".csv")]

    df_main = pd.DataFrame()

    if not excel_list:
        return pd.DataFrame(columns=expected_columns)

    for name in excel_list:
        path_excel = os.path.join(path_dir, name)
        data = pd.read_csv(
            path_excel,
            header=0,
            dtype={
                "Identifiant de la publication": str,
                "ID du compte": str,
            },
        )
        if "Heure de publication" not in data.columns and len(data.columns) == 1:
            data = pd.read_csv(
                path_excel,
                header=0,
                sep=";",
                dtype={
                    "Identifiant de la publication": str,
                    "ID du compte": str,
                },
            )
        df_main = pd.concat([df_main, data], ignore_index=True)

    if df_main.empty:
        return pd.DataFrame(columns=expected_columns)

    time_col = None
    preferred_time_cols = [
        "Heure de publication",
        "Heure de publication (UTC)",
        "Heure de publication (UTC+0)",
    ]
    for col in preferred_time_cols:
        if col in df_main.columns:
            time_col = col
            break
    if time_col is None:
        for col in df_main.columns:
            col_lower = col.lower()
            if "publication" in col_lower and ("heure" in col_lower or "date" in col_lower):
                time_col = col
                break
    if time_col is None:
        pattern = r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}"
        for col in df_main.columns:
            if df_main[col].astype(str).str.contains(pattern, na=False).any():
                time_col = col
                break
    if time_col is None:
        return pd.DataFrame(columns=expected_columns)

    df_main = df_main.fillna("0")
    df_main = df_main[df_main[time_col].astype(str).str.contains(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", na=False)]

    df_main["Date"] = pd.to_datetime(df_main[time_col], format="%m/%d/%Y %H:%M", errors="coerce")
    df_main = df_main[df_main["Date"].notna()]
    df_main["Date_only"] = df_main["Date"].dt.strftime("%Y.%m.%d")
    df_main["hour"] = df_main["Date"].dt.strftime("%H:%M:%S")
    df_main["datetime"] = df_main["Date_only"] + " " + df_main["hour"]

    rename_dict = {
        "Identifiant de la publication": "media_id",
        "Type de publication": "media_product_type",
        "Vues": "views",
        "Couverture": "reach",
        "Visites de profil": "profile_visits",
        "Partages": "shares",
        "Mentions J'aime": "likes",
        "Followers en plus": "follows",
        "Reponses": "comments",
        "RÉponses": "comments",
        "Réponses": "comments",
        "RÉPONSES": "comments",
        "RＱonses": "comments",
        "Clics sur le lien": "link clics",
        "Date_only": "date",
        "Appuis sur des stickers": "clics stickers",
        "Navigation": "navigation",
    }

    df_main = df_main.rename(columns={k: v for k, v in rename_dict.items() if k in df_main.columns})

    columns_to_drop = [
        "ID du compte",
        "Nom de profil du compte",
        "Nom du compte",
        "Description",
        "Durée (s)",
        "DurＦ (s)",
        "Duree (s)",
        "Heure de publication",
        "Permalien",
        "Commentaire des données",
        "Commentaire des donnees",
        "Commentaire des donnＦs",
        "Date",
    ]

    if time_col not in columns_to_drop:
        columns_to_drop.append(time_col)

    df_main = df_main.drop(columns=[col for col in columns_to_drop if col in df_main.columns])
    df_main["media_type"] = "STORIES"

    return df_main


if __name__ == "__main__":
    extract_stories_excel()
