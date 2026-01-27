# python -m streamlit run streamlit_commentaires.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from unidecode import unidecode
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer


# =========================
# ---------- UI -----------
# =========================

st.set_page_config(page_title="Analyse IG – Commentaires & Descriptions", layout="wide")


# =========================
# ------ Helpers ----------
# =========================

@st.cache_data(show_spinner=True)
def load_media_json(path: str) -> pd.DataFrame:
    """Charge le JSON des médias Instagram."""
    df = pd.read_json(path)
    return df


def make_stopwords() -> set:
    """Stopwords de base + ajout allemand/français usuels (adaptable)."""
    german_stopwords = {
        "aber","als","am","an","auch","auf","aus","bei","bin","bis","bist","da","damit","dann",
        "der","den","des","dem","die","das","dass","du","er","es","ein","eine","einem","einen",
        "einer","eines","für","hatte","hatten","hattest","hattet","hier","im","in","ist","ja",
        "jede","jedem","jeden","jeder","jedes","kann","können","könnte","machen","mein","meine",
        "mit","muss","musste","nach","nicht","noch","nun","oder","ohne","sehr","sein","seine",
        "sich","sie","sind","so","solche","soll","sollte","sondern","und","unser","unsere","unter",
        "vom","von","vor","war","waren","warst","was","weg","weil","weiter","welche","welchem",
        "welchen","welcher","welches","wenn","werde","werden","wie","wieder","will","wir","wird",
        "wirst","wo","wollen","wollte","würde","würden","zu","zum","zur", "ich", "isch", "us", "ha"
    }
    french_fillers = {
        "alors","avec","cest","comme","dans","des","du","elle","elles","en","est","et","ils",
        "je","la","le","les","leur","l","ma","mes","mon","ne","nos","notre","on","ou","pour",
        "plus","qu","que","qui","sans","se","ses","son","sur","ta","tes","ton","tres","très",
        "tu","un","une","vos","votre","ya","y","au","aux"
    }
    sw = set(STOPWORDS) | german_stopwords | french_fillers
    return sw


def clean_text(text: str) -> str:
    """Nettoyage léger pour analyses lexiques."""
    if not isinstance(text, str):
        return ""
    text = unidecode(text)
    text = re.sub(r"#[a-zA-Z\s]+", " ", text) # Hastag
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)            # lettres + espaces
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


TOKEN_RE = re.compile(r"\b[^\W_]+\b", flags=re.UNICODE)  # mots = lettres (accents ok), pas d'_ ni chiffres

def word_frequencies(texts: Iterable[str], stopwords: set, min_len: int = 2) -> Tuple[Counter, pd.DataFrame]:
    """Transforme une liste de textes en fréquences de mots (Counter + DataFrame) avec correspondance exacte des mots."""
    tokens: List[str] = []
    for t in texts:
        ct = clean_text(t)  # suppose que ça met en minuscules, enlève ponctuation superflue, etc.
        if not ct:
            continue
        words = [w for w in TOKEN_RE.findall(ct) if w not in stopwords and len(w) >= min_len]
        tokens.extend(words)

    freqs = Counter(tokens)
    df_freqs = pd.DataFrame(freqs.items(), columns=["mot", "frequence"]).sort_values("frequence", ascending=False)
    return freqs, df_freqs


def mean_likes_per_keyword(df: pd.DataFrame, text_col: str, likes_col: str, keywords: Iterable[str]) -> pd.DataFrame:
    """Calcule la moyenne des likes et le nombre d'occurrences par mot-clé trouvé dans text_col."""
    data: Dict[str, Dict[str, float | int]] = {}
    for kw in keywords:
        # Mot entier ; retire \b si tu veux compter les sous-chaînes.
        pattern = rf"\b{re.escape(str(kw))}\b"
        mask = df[text_col].str.contains(pattern, case=False, na=False)
        if mask.any():
            data[str(kw)] = {
                "mean_likes": float(df.loc[mask, likes_col].mean()),
                "likes_count": int(mask.sum()),
            }
    if not data:
        return pd.DataFrame(columns=["mot", "mean_likes", "count"])
    out = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "mot"})
    return out.sort_values(["mean_likes", "likes_count"], ascending=[False, False]).reset_index(drop=True)


def render_wordcloud(freqs: Counter, title: str) -> None:
    """Affiche un wordcloud Streamlit à partir d'un Counter."""
    if not freqs:
        st.info("Aucun mot à afficher.")
        return
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.pyplot(fig)


# =========================
# --------- Data ----------
# =========================

try:
    from core.paths import ORGANIC_IG_JSON_DIR
    DATA_PATH = str(ORGANIC_IG_JSON_DIR / "insights_per_media.json")
except Exception:
    DATA_PATH = "json files insta/insights_per_media.json"
data_ig_media = load_media_json(DATA_PATH)

# Colonnes attendues (défensif)
expected_cols = {
    "media_id", "text_comment", "time_comment", "id_comment", "like_count_comment",
    "replies_text_comment", "replies_like_count_comment", "replies_user_comment",
    "description_assets", "reach", "saved", "likes", "comments", "shares",
    "profile_visits", "follows", "views"
}
missing = [c for c in expected_cols if c not in data_ig_media.columns]
if missing:
    st.warning(f"Colonnes manquantes dans le JSON: {missing} — certaines fonctionnalités peuvent être limitées.")

# Table commentaires (explosion des listes)
df_comments = (
    data_ig_media
    .loc[:, [c for c in [
        "media_id", "text_comment", "time_comment", "id_comment", "like_count_comment",
        "replies_text_comment", "replies_like_count_comment", "replies_user_comment"
    ] if c in data_ig_media.columns]]
    .explode([
        c for c in [
            "text_comment", "time_comment", "id_comment", "like_count_comment",
            "replies_text_comment", "replies_like_count_comment", "replies_user_comment"
        ] if c in data_ig_media.columns
    ])
    .rename(columns={"text_comment": "comment"})
    .reset_index(drop=True)
)

STOPWORDS_ALL = make_stopwords()


# =========================
# ------- Sections --------
# =========================

st.title("🧠 Analyse des commentaires & descriptions Instagram")

tab1, tab2 = st.tabs(["📄 Descriptions", "💬 Commentaires"])


# ---------- Tab 1 : Descriptions ----------
with tab1:
    st.subheader("Analyse des descriptions & likes")

    # Sélecteur de plage de likes
    if "likes" in data_ig_media.columns:
        min_likes = int(pd.to_numeric(data_ig_media["likes"], errors="coerce").min()) if len(data_ig_media) else 0
        max_likes = int(pd.to_numeric(data_ig_media["likes"], errors="coerce").max()) if len(data_ig_media) else 0
        like_min, like_max = st.slider(
            "Filtrer par likes (descriptions)",
            min_value=min_likes, max_value=max_likes,
            value=(min_likes, max_likes),
            step=1
        )
    else:
        like_min, like_max = 0, 0
    
    # Filtre likes
    df_desc = data_ig_media.copy()
    if "likes" in df_desc.columns:
        df_desc = df_desc[
            (pd.to_numeric(df_desc["likes"], errors="coerce") >= like_min) &
            (pd.to_numeric(df_desc["likes"], errors="coerce") <= like_max)
        ]

    # KPI
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if "likes" in df_desc.columns and len(df_desc):
            st.metric("Mean likes (Non filtré)", f"{round(pd.to_numeric(data_ig_media['likes'], errors='coerce').mean(), 2)}")
        else:
            st.metric("Mean likes (Non filtré)", "—")

    # Tableau filtré
    show_cols = [c for c in ["description_assets", "reach", "likes", "comments", "shares", "follows", "views"] if c in df_desc.columns]
    st.dataframe(df_desc[show_cols] if show_cols else df_desc)


    # --- Fréquences de mots sur descriptions filtrées ---
    texts_desc = df_desc["description_assets"].dropna().astype(str).tolist() if "description_assets" in df_desc.columns else []
    freqs_desc, df_freqs_desc = word_frequencies(texts_desc, STOPWORDS_ALL, min_len=3)
    render_wordcloud(freqs_desc, "Nuage de mots – Descriptions")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Mean likes par mot-clé
        if len(data_ig_media): # df_freqs_desc
            df_mean_likes = mean_likes_per_keyword(
                df=df_desc,
                text_col="description_assets",
                likes_col="likes",
                keywords=df_freqs_desc["mot"].unique()
            )
            st.caption("Moyenne de likes par mot-clé trouvé dans les descriptions filtrées")
            st.dataframe(df_mean_likes)
            st.bar_chart(df_mean_likes, x="mot", y="mean_likes")
        else:
            st.info("Pas de mots-clés exploitables sur l'échantillon filtré.")
    with col2:
        st.caption("Fréquences des mots (descriptions filtrées)")
        st.dataframe(df_freqs_desc)
        st.bar_chart(df_freqs_desc.head(30), x="mot", y="frequence")

    


# ---------- Tab 2 : Commentaires ----------
with tab2:
    st.subheader("Analyse des commentaires & likes")

    if not len(df_comments):
        st.info("Aucun commentaire à afficher.")
    else:
        # Sélecteur plage de likes des commentaires
        if "like_count_comment" in df_comments.columns:
            # options uniques -> slider classique (plus fluide que select_slider)
            min_c = int(pd.to_numeric(df_comments["like_count_comment"], errors="coerce").min())
            max_c = int(pd.to_numeric(df_comments["like_count_comment"], errors="coerce").max())
            cmin, cmax = st.slider(
                "Filtrer par nombre de likes (commentaires)",
                min_value=min_c, max_value=max_c, value=(min_c, max_c), step=1
            )
        else:
            cmin, cmax = 0, 0

        df_comm_filtered = df_comments.copy()
        if "like_count_comment" in df_comm_filtered.columns:
            df_comm_filtered = df_comm_filtered[
                (pd.to_numeric(df_comm_filtered["like_count_comment"], errors="coerce") >= cmin) &
                (pd.to_numeric(df_comm_filtered["like_count_comment"], errors="coerce") <= cmax)
            ]

        # Affichage tableau + bar chart
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_comm_filtered)
        with c2:
            if {"id_comment", "like_count_comment"} <= set(df_comm_filtered.columns):
                st.bar_chart(df_comm_filtered, x="id_comment", y="like_count_comment")

        # Filtre texte pour les commentaires
        st.markdown("---")
        st.caption("Filtrer par chaîne dans le commentaire (après normalisation des accents)")
        filter_comment = st.text_input("Contient…", value="")
        df_comm_filtered = df_comm_filtered.copy()
        if "comment" in df_comm_filtered.columns:
            df_comm_filtered["comment_clean"] = df_comm_filtered["comment"].apply(unidecode).str.lower()
            if filter_comment:
                df_comm_filtered = df_comm_filtered[df_comm_filtered["comment_clean"].str.contains(filter_comment.lower(), na=False)]

            # Comptage par media_id (utile pour voir quels médias reviennent)
            if "media_id" in df_comm_filtered.columns:
                df_count = df_comm_filtered.groupby("media_id").size().reset_index(name="count").sort_values("count", ascending=False)
                d1, d2 = st.columns(2)
                with d1:
                    st.dataframe(df_comm_filtered[["media_id", "comment", "like_count_comment"]])
                with d2:
                    st.bar_chart(df_count, x="media_id", y="count")

            # Métriques agrégées sur médias filtrés
            if "media_id" in df_comm_filtered.columns:
                unique_ids = df_comm_filtered["media_id"].dropna().unique().tolist()
                metrics_cols = [c for c in ["media_id","media_product_type","reach","saved","likes","comments","shares","profile_visits","follows","views"] if c in data_ig_media.columns]
                df_metrics = data_ig_media[data_ig_media["media_id"].isin(unique_ids)][metrics_cols]
                if len(df_metrics):
                    df_melt = df_metrics.melt(id_vars="media_id", value_vars=[c for c in df_metrics.columns if c not in ["media_id","media_product_type"]])
                    st.bar_chart(df_melt, x="media_id", y="value", color="variable")

            # Wordcloud commentaires
            freqs_com, df_freqs_com = word_frequencies(df_comm_filtered.get("comment", pd.Series([], dtype=str)).astype(str), STOPWORDS_ALL, min_len=2)
            render_wordcloud(freqs_com, "Nuage de mots – Commentaires")
