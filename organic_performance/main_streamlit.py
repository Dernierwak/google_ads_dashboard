import os
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


class AnalysePostInsta:

    def __init__(self):
        self.data = pd.DataFrame()
        self.MAIN_DIR = Path(__file__).parent.resolve()
        try:
            from core.paths import ORGANIC_IG_JSON_DIR
            self.JSON_DIR_IMPORT = str(ORGANIC_IG_JSON_DIR)
        except Exception:
            self.JSON_DIR_IMPORT = os.path.join(self.MAIN_DIR, "json files insta")

        # état initial
        if "show_comments" not in st.session_state:
            st.session_state.show_comments = False

    def _create_dir(self):
        os.makedirs(self.JSON_DIR_IMPORT, exist_ok=True)

    @staticmethod
    @st.cache_data
    def _load_data(file_path: str):
        if os.path.exists(file_path):
            return pd.read_json(file_path)
        else:
            return pd.DataFrame()

    def _set_up_streamlit(self):
        st.set_page_config(layout="wide")

    def view_data(self):
        self._create_dir()
        self._set_up_streamlit()

        if self.data.empty:
            file_path = os.path.join(self.JSON_DIR_IMPORT, "insights_per_media.json")
            self.data = self._load_data(file_path)

        st.header("View Data")

        # --- Features calculées ---
        self.data["count_comments"] = self.data["text_comment"].apply(
            lambda x: len(x) if hasattr(x, "__len__") else 0
        )
        self.data["total_comment_likes"] = self.data["like_count_comment"].apply(
            lambda x: sum(x) if isinstance(x, (list, tuple)) else 0
        )
        self.data["max_like_per_comment"] = self.data["like_count_comment"].apply(
            lambda x: (max(x) if isinstance(x, (list, tuple)) and len(x) > 0 else 0)
        )
        self.data["avg_likes_per_comment"] = self.data["like_count_comment"].apply(
            lambda v: round(sum(v) / len(v), 2) if isinstance(v, (list, tuple)) and len(v) > 0 else 0
        )

        data_comments = self.data[[
            "description_assets",
            "count_comments",
            "total_comment_likes",
            "max_like_per_comment",
            "avg_likes_per_comment"
        ]].copy()

        # --- Filtres ---
        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            min_comment = int(data_comments["count_comments"].min()) if not data_comments.empty else 0
            max_comment = int(data_comments["count_comments"].max()) if not data_comments.empty else 0
            st.slider("Nombre de commentaires",
                      min_value=min_comment, max_value=max_comment,
                      value=(min_comment, max_comment), step=1, key="count_comments")

        with col_2:
            min_total_likes = int(data_comments["total_comment_likes"].min()) if not data_comments.empty else 0
            max_total_likes = int(data_comments["total_comment_likes"].max()) if not data_comments.empty else 0
            st.slider("Total de likes (tous les commentaires)",
                      min_value=min_total_likes, max_value=max_total_likes,
                      value=(min_total_likes, max_total_likes), step=1, key="total_comment_likes")

        with col_3:
            min_max_like = int(data_comments["max_like_per_comment"].min()) if not data_comments.empty else 0
            max_max_like = int(data_comments["max_like_per_comment"].max()) if not data_comments.empty else 0
            st.slider("Max likes sur un commentaire",
                      min_value=min_max_like, max_value=max_max_like,
                      value=(min_max_like, max_max_like), step=1, key="max_like_per_comment")

        with col_4:
            min_avg_like = int(data_comments["avg_likes_per_comment"].min()) if not data_comments.empty else 0
            max_avg_like = int(data_comments["avg_likes_per_comment"].max()) if not data_comments.empty else 0
            st.slider("Moyenne de likes par commentaire",
                      min_value=min_avg_like, max_value=max_avg_like,
                      value=(min_avg_like, max_avg_like), step=1, key="avg_likes_per_comment")

        # masque
        filter_mask = pd.Series(True, index=data_comments.index)
        if "count_comments" in st.session_state:
            min_c, max_c = st.session_state.count_comments
            filter_mask &= data_comments["count_comments"].between(min_c, max_c)
        if "total_comment_likes" in st.session_state:
            min_l, max_l = st.session_state.total_comment_likes
            filter_mask &= data_comments["total_comment_likes"].between(min_l, max_l)
        if "max_like_per_comment" in st.session_state:
            min_ml, max_ml = st.session_state.max_like_per_comment
            filter_mask &= data_comments["max_like_per_comment"].between(min_ml, max_ml)
        if "avg_likes_per_comment" in st.session_state:
            min_al, max_al = st.session_state.avg_likes_per_comment
            filter_mask &= data_comments["avg_likes_per_comment"].between(min_al, max_al)

        st.data_editor(data_comments[filter_mask], use_container_width=True)

        # ---- Boutons persistants via session_state ----
        # Un bouton pour afficher/masquer
        cols_btn = st.columns(2)
        with cols_btn[0]:
            if st.button("Voir les commentaires", key="btn_show"):
                st.session_state.show_comments = True
        with cols_btn[1]:
            if st.button("Masquer", key="btn_hide"):
                st.session_state.show_comments = False

        # ---- Zone commentaires (persiste tant que show_comments=True) ----
        if st.session_state.show_comments:
            index_true = data_comments.index[filter_mask]

            filter_index = st.multiselect(
                "Sélectionne les lignes (indices) à détailler",
                options=list(index_true),
                default=list(index_true),    # garde tout par défaut
                key="comment_rows"           # persiste la sélection également
            )

            if not filter_index:
                st.info("Aucune ligne sélectionnée.")
                return

            cols = ["media_id", "permalink", "text_comment", "like_count_comment", "replies_text_comment"]
            data_c = self.data.loc[filter_index, cols].copy()
            data_c = data_c.explode(["text_comment", "like_count_comment", "replies_text_comment"])

            st.data_editor(data_c, use_container_width=True)


def main():
    analyser = AnalysePostInsta()
    analyser.view_data()


if __name__ == "__main__":
    main()
