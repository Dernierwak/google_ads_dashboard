import pandas as pd
import streamlit as st
import os
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score

from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Google Ads & Meta Ads - Analyse des Termes de Recherche",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitGADS():

    def __init__(self):
        self.MAIN_DIR = Path(__file__).parent.resolve()
        self.PATH_DIR = f"{self.MAIN_DIR}/import_dir"
        self.EXCEL_LIST = None
        self.dashboard = ""
    
    def _create_dir(self):
        os.makedirs(self.PATH_DIR, exist_ok=True)

    def _find_excel(self):
        self._create_dir()
        files = os.listdir(self.PATH_DIR)
        if files:
            self.EXCEL_LIST = [x for x in files if x.endswith(".xlsx")]
        else:
            st.warning(f"Add the data in the Excel Dir: {self.PATH_DIR}")

    @st.cache_data(show_spinner=False)
    def _load_data(_self, excel_name, start_index=2) -> pd.DataFrame | None:
        if excel_name is None:
            return None

        try:
            with st.spinner("Load data.."):
                data = pd.read_excel(
                    os.path.join(str(_self.PATH_DIR), excel_name),
                    header=start_index
                )

                if "purchase_True" in data.columns:
                    data["purchase_True"] = (
                        data["purchase_True"]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                        .astype(float)
                    )

                # Conversion de type sécurisée
                type_mapping = {
                    "Clics": int,
                    "Impr.": int,
                    "CTR": float,
                    "CPC moy.": float,
                    "Coût": float,
                    "Toutes les conversions": float,
                }

                for col, dtype in type_mapping.items():
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype(dtype)

                return data
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    def _dashboard_setup(self):
        self._find_excel()
        with st.sidebar:
            self.EXCEL = st.selectbox(
                "Choose excel file",
                self.EXCEL_LIST,
                key="excel_choosed",
                placeholder="No files",
                index=None
            )

            # Sélection de l'index de ligne pour le header
            st.session_state["header_index"] = st.number_input(
                "Ligne de début (header)",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
                key="header_row_input",
                help="Ligne où commence les données (0 = première ligne)"
            )
    
    def _rename_cols(self, df) -> pd.DataFrame:
        with st.expander("Remap Col Name"):
            st.caption("Mapper les colonnes de votre fichier aux noms standards")

            # Section 1: Colonnes de campagne/adset
            st.markdown("**Colonnes de structure**")
            flex1 = st.container(
                horizontal=True,
                horizontal_alignment="left",
                gap="small"
            )
            campaign = flex1.text_input("Campaign", placeholder="Campaign name")
            adset = flex1.text_input("Adset", placeholder="Ad set / Ad group")

            st.divider()

            # Section 2: Colonnes de métriques
            st.markdown("**Colonnes de métriques**")
            flex2 = st.container(
                horizontal=True,
                horizontal_alignment="left",
                gap="small"
            )

            impr = flex2.text_input("Impression", placeholder="Impressions")
            clics = flex2.text_input("Clics", placeholder="Clicks")
            ctr = flex2.text_input("CTR", placeholder="CTR")

            flex3 = st.container(
                horizontal=True,
                horizontal_alignment="left",
                gap="small"
            )

            cpc = flex3.text_input("CPC moy.", placeholder="Avg CPC")
            cost = flex3.text_input("Coût", placeholder="Cost/Spend")
            conv_targ = flex3.text_input("Taux conv target", placeholder="Conv. rate")

            mapping = {
                campaign: "campaign",
                adset: "adset",
                impr: "Impr.",
                clics: "Clics",
                ctr: "CTR",
                cpc: "CPC moy.",
                cost: "Coût",
                conv_targ: "taux conv target"
            }

            col_rename = {k: v for k, v in mapping.items() if k}

            if col_rename:
                df = df.rename(columns=col_rename)
                st.success(f"✅ {len(col_rename)} colonnes renommées")

        return df


    def _apply_iqr_filter(self, df, columns):
        """
        Filtre les valeurs aberrantes avec la méthode IQR (Interquartile Range)
        
        Parameters:
        -----------
        df : DataFrame
            Les données à filtrer
        columns : list
            Liste des colonnes sur lesquelles appliquer le filtre IQR
            
        Returns:
        --------
        DataFrame : Les données filtrées sans valeurs aberrantes
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        nb_initial = len(df_clean)
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            # Convertir en numérique
            series = pd.to_numeric(df_clean[col], errors="coerce")
            
            # Si que des NaN, on passe
            if series.dropna().empty:
                continue
            
            # Calculer les quartiles
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            # Si pas de variabilité, on passe
            if iqr == 0:
                st.info(f"ℹ️ {col}: Pas de variabilité (IQR = 0), pas de filtrage")
                continue

            # Calculer les limites
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Compter les valeurs hors limites
            outliers = ((series < lower_bound) | (series > upper_bound)) & series.notna()
            n_outliers = outliers.sum()

            st.write(f"**{col}**: [{lower_bound:.2f}, {upper_bound:.2f}] - {n_outliers} valeurs aberrantes détectées")
            
            # Filtrer : garder les valeurs dans les limites OU les NaN
            mask = ((series >= lower_bound) & (series <= upper_bound)) | series.isna()
            df_clean = df_clean[mask]
        
        nb_final = len(df_clean)
        nb_removed = nb_initial - nb_final
        
        if nb_removed > 0:
            pct_removed = (nb_removed / nb_initial) * 100
            st.success(f"✅ Filtre IQR: {nb_removed} lignes supprimées ({pct_removed:.1f}%)")
        else:
            st.info("ℹ️ Filtre IQR: Aucune valeur aberrante détectée")
        
        return df_clean

    def _create_slider(self, col, data_filtered, label, is_percentage=False, is_integer=False):
        """
        Crée un slider pour filtrer les données sur une colonne spécifique

        Parameters:
        -----------
        col : str
            Nom de la colonne
        data_filtered : DataFrame
            Données à filtrer
        label : str
            Label du slider
        is_percentage : bool
            Si True, retire le symbole % avant conversion
        is_integer : bool
            Si True, convertit en int, sinon en float

        Returns:
        --------
        Series : Masque booléen pour filtrer les données
        """
        if col not in data_filtered.columns:
            return pd.Series(True, index=data_filtered.index)

        series = data_filtered[col].astype(str)

        if is_percentage:
            series = series.str.replace("%", "", regex=False)

        series = series.str.replace(",", ".", regex=False)
        series = pd.to_numeric(series, errors="coerce")

        if series.dropna().empty:
            st.info(f"Aucune donnée valide pour {label}")
            return pd.Series(True, index=data_filtered.index)

        if is_integer:
            min_v, max_v = int(series.min()), int(series.max())
            step = 1
        else:
            min_v, max_v = float(series.min()), float(series.max())
            step = None

        min_s, max_s = st.slider(
            label,
            min_value=min_v,
            max_value=max_v,
            value=(min_v, max_v),
            step=step
        )

        return (series >= min_s) & (series <= max_s)

    def main_dash(self, col_campaign="campaign", col_adset="adset"):
        self._dashboard_setup()

        # Récupérer l'index du header
        header_idx = st.session_state.get("header_index", 2)
        data = self._load_data(st.session_state.get("excel_choosed"), start_index=header_idx)

        st.title("📊 Dashboard Google Ads & Meta Ads")
        st.caption("Analyse des performances et clustering des termes de recherche")

        self.dashboard = st.selectbox(
            "Plateforme publicitaire",
            options=["Meta", "GADS"],
            help="Sélectionnez la plateforme pour adapter l'analyse"
        )

        if data is not None and not data.empty:
            # IMPORTANT: récupérer le DataFrame renommé
            data = self._rename_cols(data)
            with st.sidebar:
                st.header("📋 Filtres de campagne")

                selected_campaign = None
                selected_adset = None

                # Filtre Campagne
                if col_campaign in data.columns:
                    campaigns = data[col_campaign].dropna().unique().tolist()
                    campaigns.sort()

                    selected_campaign = st.selectbox(
                        "Campagne",
                        options=[None] + campaigns,
                        format_func=lambda x: "Toutes les campagnes" if x is None else x,
                        key="selected_campaign",
                        help=f"{len(campaigns)} campagnes disponibles"
                    )
                else:
                    st.warning("⚠️ Colonne 'campaign' non trouvée dans vos données")

                # Filtre Adset (dépend de la campagne sélectionnée)
                if col_adset in data.columns:
                    # Filtrer les adsets en fonction de la campagne sélectionnée
                    if selected_campaign and col_campaign in data.columns:
                        adsets = data[data[col_campaign] == selected_campaign][col_adset].dropna().unique().tolist()
                    else:
                        adsets = data[col_adset].dropna().unique().tolist()

                    adsets.sort()

                    selected_adset = st.selectbox(
                        "Adset / Groupe d'annonces",
                        options=[None] + adsets,
                        format_func=lambda x: "Tous les adsets" if x is None else x,
                        key="selected_adset",
                        help=f"{len(adsets)} adsets disponibles"
                    )
                else:
                    st.warning("⚠️ Colonne 'adset' non trouvée dans vos données")

                col_convs = st.text_input(
                    "Colonne des conversions cible",
                    key="conv_target",
                    placeholder="Ex: purchase, lead, signup..."
                )

                # --- FILTRE IQR ---
                st.divider()
                st.header("🔍 Filtre IQR")
                apply_iqr = st.checkbox(
                    "Appliquer le filtre IQR (valeurs aberrantes)",
                    value=False,
                    key="apply_iqr_filter",
                    help="Supprime les valeurs aberrantes selon la méthode IQR (Interquartile Range)"
                )

            # --- Build mask ---
            mask = pd.Series(True, index=data.index)

            # Campaign filter
            if selected_campaign:
                mask &= data[col_campaign] == selected_campaign

            # Adset filter
            if selected_adset:
                mask &= data[col_adset] == selected_adset

            # Conversions filter
            if col_convs:
                if col_convs not in data.columns:
                    st.warning(f"⚠️ Colonne '{col_convs}' non trouvée")
                else:
                    mask &= data[col_convs] > 0
                    # Éviter division par zéro
                    data["taux conv target"] = data.apply(
                        lambda row: (row[col_convs] * 100 / row["Clics"]) if row["Clics"] > 0 else 0,
                        axis=1
                    )

            # Appliquer le masque campaign/adset/conversions
            data_filtered = data.loc[mask].copy()

            # --- APPLIQUER LE FILTRE IQR ---
            if apply_iqr:
                with st.expander("📊 Détails du filtrage IQR", expanded=True):
                    data_filtered = self._apply_iqr_filter(data_filtered, ["Impr.", "Clics"])

            # --- Maintenant les sliders utilisent data_filtered ---
            st.subheader("🎚️ Filtres de métriques")
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            # Créer un nouveau masque pour les sliders
            mask_sliders = pd.Series(True, index=data_filtered.index)

            with col1:
                mask_sliders &= self._create_slider("Impr.", data_filtered, "Impressions", is_integer=True)

            with col2:
                mask_sliders &= self._create_slider("Clics", data_filtered, "Clics", is_integer=True)

            with col3:
                mask_sliders &= self._create_slider("CTR", data_filtered, "CTR (%)", is_percentage=True)

            with col4:
                mask_sliders &= self._create_slider("CPC moy.", data_filtered, "CPC moyen")

            with col5:
                mask_sliders &= self._create_slider("Coût", data_filtered, "Coût")

            with col6:
                if "taux conv target" in data_filtered.columns:
                    mask_sliders &= self._create_slider("taux conv target", data_filtered, "Taux conv target (%)", is_percentage=True)
                else:
                    st.info("Choisir la colonne conv target")
            
            # Appliquer le masque des sliders
            data_selected = data_filtered.loc[mask_sliders].copy()

            if data_selected.empty:
                st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés. Ajustez vos critères.")
            else:
                # Sauvegarder dans session_state
                st.session_state["data_selected"] = data_selected

                # Afficher les métriques clés
                st.subheader(f"📊 Résumé des performances ({len(data_selected)} lignes)")
                metric_cols = st.columns(5)

                with metric_cols[0]:
                    total_impr = data_selected["Impr."].sum() if "Impr." in data_selected.columns else 0
                    st.metric("Impressions totales", f"{total_impr:,.0f}")

                with metric_cols[1]:
                    total_clics = data_selected["Clics"].sum() if "Clics" in data_selected.columns else 0
                    st.metric("Clics totaux", f"{total_clics:,.0f}")

                with metric_cols[2]:
                    avg_ctr = data_selected["CTR"].mean() if "CTR" in data_selected.columns else 0
                    st.metric("CTR moyen", f"{avg_ctr:.2f}%")

                with metric_cols[3]:
                    total_cost = data_selected["Coût"].sum() if "Coût" in data_selected.columns else 0
                    st.metric("Coût total", f"{total_cost:,.2f} CHF")

                with metric_cols[4]:
                    if "taux conv target" in data_selected.columns:
                        avg_conv = data_selected["taux conv target"].mean()
                        st.metric("Taux conv. moyen", f"{avg_conv:.2f}%")
                    else:
                        st.metric("Conversions", "N/A")

                st.divider()

                # Afficher les données
                col_title, col_export = st.columns([4, 1])
                with col_title:
                    st.subheader("📋 Données détaillées")
                with col_export:
                    csv = data_selected.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Exporter CSV",
                        data=csv,
                        file_name=f"export_filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Télécharger les données filtrées au format CSV"
                    )

                st.data_editor(data_selected, key="main_data_editor", height=400)

        else:
            st.info("📁 Veuillez sélectionner un fichier Excel dans la barre latérale pour commencer")

    def general_graph(self):
        """Affiche un graphique combiné scatter + bar"""

        # Vérifier que les données existent
        if "data_selected" not in st.session_state:
            st.info("ℹ️ Veuillez d'abord sélectionner des données dans la section principale.")
            return

        data_selected = st.session_state["data_selected"]

        if data_selected.empty:
            st.warning("⚠️ Aucune donnée à afficher")
            return

        # Vérifier que les colonnes nécessaires existent
        required_cols = ["Impr.", "Clics", "Coût"]
        missing_cols = [col for col in required_cols if col not in data_selected.columns]

        if missing_cols:
            st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
            return
        
        # Choisir la métrique pour la taille des points
        list_sized = [
            col for col in ["CTR", "taux conv target"]
            if col in data_selected.columns
        ]
        
        if not list_sized:
            st.error("Aucune colonne disponible pour la taille des points")
            return
        
        sized_col = st.selectbox("Choisir la métrique pour la taille", options=list_sized)

        # Créer le graphique
        fig = go.Figure()

        # Scatter: Impr vs Clicks, taille = CTR ou Taux conv
        fig.add_trace(go.Scatter(
            x=data_selected["Impr."],
            y=data_selected["Clics"],
            mode="markers",
            name="Impr vs Clicks",
            marker=dict(
                size=data_selected[sized_col],
                sizemode="area",
                sizeref=data_selected[sized_col].max() / 40 if data_selected[sized_col].max() > 0 else 1,
                showscale=True,
                colorscale="Viridis"
            ),
            text=[f"Impr: {i}<br>Clics: {c}<br>{sized_col}: {s:.2f}" 
                  for i, c, s in zip(data_selected["Impr."], 
                                     data_selected["Clics"], 
                                     data_selected[sized_col])],
            hovertemplate="%{text}<extra></extra>"
        ))

        # Bar: Coût
        fig.add_trace(go.Bar(
            x=data_selected["Impr."],
            y=data_selected["Coût"],
            name="Coût",
            opacity=0.5,
            marker_color="lightcoral"
        ))

        fig.update_layout(
            title="📈 Vue Générale: Impressions vs Clics et Coût",
            xaxis_title="Impressions",
            yaxis_title="Valeur",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------ SBERT cache helpers ------------------
    @st.cache_resource
    def _sbert_model(_self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        return SentenceTransformer(model_name)

    @st.cache_data(show_spinner=False)
    def _embed_texts(_self, texts:tuple, model_name:str) -> np.ndarray:
        model = _self._sbert_model(model_name)
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def cluster(self):
        with st.sidebar:
            st.header("Paramètres du Clustering")
            list_model = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]

            model_name = st.selectbox(
                "Choisir le modèle d'embedding",
                list_model,
                key="sbert_model_name",
                help="Modèle pour vectoriser les termes de recherche"
            )
            auth_n_cluster = st.checkbox(
                "Détection automatique du nombre de clusters",
                help="Utilise le score silhouette pour trouver le nombre optimal de clusters (2-20)"
            )
            active_cluster = st.button("🚀 Lancer le clustering", type="primary")
        
        data_cluster = st.session_state.get("data_selected")
        
        # ✅ Vérification que data_selected existe
        if data_cluster is None or data_cluster.empty:
            st.warning("⚠️ Aucune donnée sélectionnée pour le clustering")
            return
        
        standart_cluster_col = "Terme de recherche"
        texts = None

        if standart_cluster_col in data_cluster.columns:
            st.success(f"✅ Colonne de clustering trouvée: {standart_cluster_col}")
            texts = (
                data_cluster[standart_cluster_col]
                .fillna("")
                .str.strip()
            )
        else:
            st.info(f"ℹ️ Colonne standard '{standart_cluster_col}' non trouvée")
            col_cluster = st.text_input(
                "Nom de la colonne à utiliser pour le clustering",
                placeholder="Ex: Search term, Keyword, etc."
            )
            if col_cluster:
                if col_cluster in data_cluster.columns:
                    texts = (
                        data_cluster[col_cluster]
                        .fillna("")
                        .str.strip()
                    )
                    st.success(f"✅ Colonne '{col_cluster}' sélectionnée")
                else:
                    st.error(f"❌ Colonne '{col_cluster}' introuvable dans les données")
            else:
                st.warning("⚠️ Veuillez spécifier une colonne pour le clustering")

        if active_cluster and texts is not None:
            with st.spinner("Clustering en cours..."):
                emb = self._embed_texts(texts=tuple(texts), model_name=model_name)
                n_samples = emb.shape[0]
                if n_samples < 2:
                    st.warning("Pas assez de lignes pour lancer le clustering (minimum 2).")
                    return

                k_vals = 5
                best_k = None
                best_s = None

                if auth_n_cluster:
                    if n_samples < 3:
                        st.warning("Pas assez de lignes pour la detection automatique. Passez en manuel.")
                        best_k = min(2, n_samples)
                        labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(emb)
                    else:
                        best_k, best_s = None, -1
                        max_k = min(20, n_samples - 1)
                        for k in range(2, max_k + 1):
                            labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
                            s = silhouette_score(emb, labels=labels, metric="cosine")
                        
                            if s > best_s:
                                best_s, best_k = s, k

                        st.caption(f"Le meilleur k est {best_k}\nLe score silhouette est {best_s:.3f}")
                        labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(emb)

                else:
                    # ? Correction: d‚finir best_k et best_s aussi pour le cas non-automatique
                    best_k = min(k_vals, n_samples)
                    labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(emb)
                    if n_samples > best_k:
                        best_s = silhouette_score(emb, labels=labels, metric="cosine")

                data_cluster["cluster"] = labels

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre de clusters", best_k)
                with col2:
                    if best_s is None:
                        st.metric("Score Silhouette", "N/A")
                    else:
                        st.metric("Score Silhouette", f"{best_s:.3f}")

                st.data_editor(data_cluster)
                st.session_state["df_cluster"] = data_cluster

    def cluster_barplot(self):
        if "df_cluster" in st.session_state:

            data_cluster = st.session_state["df_cluster"]

            agg_dict = {
                "Terme de recherche": "count",
                "Coût": "sum",
                "Impr.": "sum",
                "Clics": "sum",
                "CTR": "mean",
                "Toutes les conversions": "sum",
                "CPC moy.": "mean",
                "Coût/conv.": "mean"
            }

            if "taux conv target" in data_cluster.columns:
                agg_dict["taux conv target"] = "mean"

            group_by_cluster = data_cluster.groupby("cluster").agg(agg_dict).reset_index()
            
            numeric_cols = [col for col in agg_dict.keys()]
            
            scaled = group_by_cluster.copy()
            scaler = MinMaxScaler(feature_range=(0.1, 0.99))
            scaled[numeric_cols] = scaler.fit_transform(group_by_cluster[numeric_cols])

            scaled_melted = pd.melt(
                scaled, 
                id_vars=["cluster"], 
                value_vars=numeric_cols,
                var_name="metric", 
                value_name="scaled_value"
            )
            
            # Affichage
            #st.subheader("📊 Données agrégées par cluster (scaled 0-1)")
            #st.data_editor(scaled_melted)
            st.session_state["scaled"] = scaled
            #st.data_editor(scaled)
            st.subheader("📊 Barplot comparatif des clusters")
            st.bar_chart(scaled_melted, x="cluster", y="scaled_value", color="metric", stack=False)

    def perf_barplot(self):
        if "scaled" not in st.session_state or "df_cluster" not in st.session_state:
            st.info("ℹ️ Lancez d'abord le clustering et le cluster_barplot pour voir les performances.")
            return

        df_cluster = st.session_state["scaled"].copy()      # agrégé par cluster + scaled
        data_cluster = st.session_state["df_cluster"].copy()  # lignes originales + cluster

        if df_cluster.empty or data_cluster.empty:
            st.warning("⚠️ Données de clustering vides")
            return

        # --- Scores (tout est déjà normalisé 0-1) ---
        df_cluster["perf_opportunite_visibilite"] = (
            df_cluster["CTR"]
            * (1 - df_cluster["Impr."])
            * (1 - df_cluster["CPC moy."])
        ).round(4)

        score_cols = ["perf_opportunite_visibilite"]

        if "taux conv target" in df_cluster.columns:
            df_cluster["perf_top_performance"] = (
                df_cluster["CTR"]
                * df_cluster["taux conv target"]
                * (1 - df_cluster["CPC moy."])
            ).round(4)

            df_cluster["perf_potentiel_scaling"] = (
                (1 - df_cluster["CTR"])
                * df_cluster["taux conv target"]
                * (1 - df_cluster["CPC moy."])
            ).round(4)

            df_cluster["perf_risque"] = (
                (1 - df_cluster["CTR"])
                * (1 - df_cluster["taux conv target"])
                * df_cluster["CPC moy."]
            ).round(4)

            score_cols += [
                "perf_top_performance",
                "perf_potentiel_scaling",
                "perf_risque",
            ]

        # --- Optionnel: trier pour avoir les tabs dans un ordre utile ---
        df_cluster = df_cluster.sort_values("perf_opportunite_visibilite", ascending=False).reset_index(drop=True)

        st.subheader("📊 Scores de performance par cluster")
        st.bar_chart(df_cluster, x="cluster", y=score_cols, stack=False)

        # --- Tabs: 1 tab = 1 cluster ---
        tab_labels = [
            f"Cluster {int(row['cluster'])} | Opp {row['perf_opportunite_visibilite']:.2f}"
            for _, row in df_cluster.iterrows()
        ]
        tabs = st.tabs(tab_labels)

        for tab, (_, row) in zip(tabs, df_cluster.iterrows()):
            cluster_id = row["cluster"]

            with tab:
                st.subheader(f"Cluster {cluster_id}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Opportunité visibilité", float(row["perf_opportunite_visibilite"]))
                col2.metric("Top performance", float(row.get("perf_top_performance", 0)))
                col3.metric("Risque", float(row.get("perf_risque", 0)))

                st.caption("📌 Agrégé (scaled)")
                st.dataframe(row.to_frame().T, use_container_width=True)

                st.caption("🔎 Lignes du cluster (données brutes filtrées)")
                df_lines = data_cluster[data_cluster["cluster"] == cluster_id].copy()
                st.dataframe(df_lines, use_container_width=True)

            
if __name__ == "__main__":
    dashboard = StreamlitGADS()
    dashboard.main_dash()
    st.divider()
    dashboard.general_graph()
    dashboard.cluster()
    dashboard.cluster_barplot()
    dashboard.perf_barplot()