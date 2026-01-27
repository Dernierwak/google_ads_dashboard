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


class StreamlitGADS():

    def __init__(self):
        self.MAIN_DIR = Path(__file__).parent.resolve()
        self.PATH_DIR = Path(self.MAIN_DIR) / "import_dir"
        self.FILES_LIST = None
        self.data_index_start = 2
        self.FILE_TYPE = None
        self.TYPE_DASHBOARD = None
        self.RENAME_COLS = dict
        self.list_sum_col = ["Impr.", "Clics", "Coût"]
        self.list_mean_col = ["CTR", "CPC moy."]
        self.list_base_col = ["campaign", "ads_id", "adset"]

    def _create_dir(self):
        os.makedirs(self.PATH_DIR, exist_ok=True)

    def _find_files(self):
        self._create_dir()
        with st.sidebar:
            with st.expander("File Parametter"):
                self.data_index_start = st.number_input("change the index of the dataframe if needed", min_value=0, max_value=10, step=1)
                self.TYPE_DASHBOARD = st.pills("Choose the type of Dashboard", options=["GADS", "META"], default="GADS")
                file_type = st.pills("File Type", options=["Excel", "json"])
                
                if os.listdir(self.PATH_DIR) != None:
                    if file_type == "Excel":
                        self.FILES_LIST = [x for x in os.listdir(self.PATH_DIR) if x.endswith(".xlsx")]
                    else:
                        self.FILES_LIST = [x for x in os.listdir(self.PATH_DIR) if x.endswith(".json")]
                else:
                    st.markdown(f"Add the data in the Excel Dir: {self.PATH_DIR}")
            
                self.FILE_TYPE = file_type

    def _rename_cols(_self, df:pd.DataFrame) -> pd.DataFrame:
        with st.sidebar:
            with st.expander("Remap Col Name"):
                st.markdown(f"The cols Name:\n{df.columns.tolist()}")
                flex = st.container(
                    horizontal=True,
                    horizontal_alignment="left",
                    gap="small"
                )
                kw = flex.text_input("KW")
                impr = flex.text_input("Impression")
                clics = flex.text_input("Clics")
                ctr = flex.text_input("CTR")
                cpc = flex.text_input("CPC moy.")
                cost = flex.text_input("Coût")
                campaigns_col = flex.text_input("Campaign_name")
                adsets_col = flex.text_input("Adsets_name")
                col_convs = flex.text_input("Conversions column", key="conv_target")

                mapping = {
                    kw: "kw",
                    impr: "Impr.",
                    clics: "Clics",
                    ctr: "CTR",
                    cpc: "CPC moy.",
                    cost: "Coût",
                    campaigns_col: "campaign",
                    adsets_col: "adset",
                }

                if col_convs:
                    _self.list_sum_col.append(col_convs)
                    mapping["conv_target"] = col_convs #mapping.update({"conv_target": col_convs})
                
                st.session_state.list_total_col = _self.list_base_col +_self.list_sum_col + _self.list_mean_col
                missing = [k for k in st.session_state.list_total_col if k not in mapping.values()]
                if missing:
                    mapping.update({k: k for k in missing})
                    st.markdown(f"Missing cols added: {missing}")

                col_rename = {k: v for k, v in mapping.items() if k}

                if col_rename:
                    df = df.rename(columns=col_rename)
                
                with st.expander("Add news cols"):
                    st.info("use the ',' to split")
                    flex = st.container(
                        horizontal=True,
                        horizontal_alignment="left",
                        gap="small"
                    )

                    sum_col_add = flex.text_input("Sum cols")
                    mean_col_add = flex.text_input("Mean cols")
                    stand_col_add = flex.text_input("Standart Col")

                    if sum_col_add:
                        cols = [c.strip() for c in sum_col_add.split(",") if c.strip()]
                        _self.list_sum_col.extend(cols)

                    if mean_col_add:
                        _self.list_mean_col.extend([c.strip() for c in mean_col_add.split(",") if c.strip()])

                    if stand_col_add:
                        _self.list_base_col.extend([c.strip() for c in stand_col_add.split(",") if c.strip()])

            #st.markdown(mapping.items())
            #st.markdown(_self.list_sum_col)

        _self.RENAME_COLS = col_rename
        return df

    @st.cache_data(show_spinner=False)
    def _load_data(_self, excel_name, start_index) -> pd.DataFrame:
        st.set_page_config(page_title="Google Ads - Search Termes Analyse", layout="wide")
        if excel_name is None:
            return None

        with st.spinner("Load data.."):
            if excel_name.endswith(".xlsx"):
                data = pd.read_excel(
                    os.path.join(str(_self.PATH_DIR), excel_name),
                    header=start_index
                )
            
            else:
                data = pd.read_json(
                    os.path.join(str(_self.PATH_DIR), excel_name)
                )

            return data

    def _postprocess_data(_self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None:
            return None

        if "purchase_True" in data.columns:
            data["purchase_True"] = (
                data["purchase_True"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

        type_map = {
            "Clics": int,
            "Impr.": int,
            "CTR": float,
            "CPC moy.": float,
            "Co–t": float,
            "Toutes les conversions": float,
        }

        cols = {k: v for k, v in type_map.items() if k in data.columns}
        
        if cols:
            data[list(cols.keys())] = (
                data[list(cols.keys())]
                .replace("--", "0")
                .astype(cols)
            )

        return data

    def _dashboard_setup(self):
        self._find_files()
        with st.sidebar:
            st.selectbox(
                "Choose excel file", 
                self.FILES_LIST, 
                key="excel_choosed", 
                placeholder="No files", 
                index=None
            )

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
                st.info(f"ℹ️ {col}: IQR = 0, pas de filtrage")
                continue
            
            # Calculer les limites
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            st.write(f"**{col}**: Limites [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Filtrer : garder les valeurs dans les limites OU les NaN
            mask = ((series >= lower_bound) & (series <= upper_bound)) | series.isna()
            df_clean = df_clean[mask]
        
        nb_final = len(df_clean)
        nb_removed = nb_initial - nb_final
        
        if nb_removed > 0:
            st.success(f"✅ Filtre IQR: {nb_removed} lignes supprimées")
        else:
            st.info("ℹ️ Filtre IQR: Aucune valeur aberrante détectée")
        
        return df_clean

    def main_dash(self, col_campaign="campaign", col_adset="adset"):
        self._dashboard_setup()
        #st.markdown(st.session_state)
        start_index = self.data_index_start
        data = self._load_data(st.session_state.get("excel_choosed"), start_index=start_index)

        if data is not None:
                data = self._rename_cols(data)
                data = self._postprocess_data(data)
                if self.TYPE_DASHBOARD == "META":
                    cols = st.session_state.list_total_col
                    #st.markdown(cols)
                    #st.markdown(data.columns)
                    data = data[cols]

                    dict_col = {}
                    dict_col.update({k: "sum" for k in self.list_sum_col if k in self.RENAME_COLS.values()})
                    dict_col.update({k: "mean" for k in self.list_mean_col if k in self.RENAME_COLS.values()})
                    dict_col.update({k: "first" for k in self.list_base_col if k in self.RENAME_COLS.values()})
                    data = data.groupby(by="ads_id").agg(dict_col)

        if data is not None and not data.empty:
            with st.sidebar:
                st.header("Choose the campaigns / adsets")

                selected_campaign = None
                selected_adset = None

                if col_campaign in data.columns:
                    selected_campaign = st.multiselect(
                        "Choose the campaign",
                        data[col_campaign].dropna().unique(),
                        key="selected_campaign",
                        placeholder="Choose a campaign"
                    )
                else:
                    st.warning("Your excel doesn't have campaign column, remap it")

                if col_adset in data.columns:
                    selected_adset = st.text_input(
                        "Choose the adset"
                    )
                else:
                    st.warning("Your excel doesn't have adset column, remap it")



                # --- FILTRE IQR ---
                st.divider()
                st.header("Filtre IQR")
                with st.expander("The IQR Parametter"):
                    apply_iqr = st.checkbox("Appliquer le filtre IQR", value=False, key="apply_iqr_filter")

            # --- Build mask ---
            mask = pd.Series(True, index=data.index)

            # Campaign filter
            if selected_campaign:
                mask &= data[col_campaign].isin(selected_campaign)

            # Adset filter
            if selected_adset:
                mask &= data[col_adset].str.lower().str.contains(selected_adset)

            # Conversions filter
            if "conv_target" in st.session_state:
                if st.session_state.conv_target not in data.columns:
                    st.warning("Conversions column not found")
                else:
                    mask &= data[st.session_state.conv_target] >= 0
                    data["taux conv target"] = np.where(
                        data["Clics"] > 0,
                        data[st.session_state.conv_target] * 100 / data["Clics"],
                        0,
                    )

            # Appliquer le masque campaign/adset/conversions
            data_filtered = data.loc[mask].copy()

            # --- APPLIQUER LE FILTRE IQR ---
            if apply_iqr:
                with st.expander("📊 Détails du filtrage IQR", expanded=True):
                    data_filtered = self._apply_iqr_filter(data_filtered, ["Impr.", "Clics"])

            # --- Maintenant les sliders utilisent data_filtered ---
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            # Créer un nouveau masque pour les sliders
            mask_sliders = pd.Series(True, index=data_filtered.index)

            with col1:
                # Impressions
                if "Impr." in data_filtered.columns:
                    impr = pd.to_numeric(
                        data_filtered["Impr."].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce"
                    )

                    if not impr.dropna().empty:
                        min_v, max_v = int(impr.min()), int(impr.max())
                        min_s, max_s = st.slider(
                            "Impressions",
                            min_value=min_v if min_v > 0 else min_v + 1,
                            max_value=max_v,
                            value=(min_v, max_v),
                            step=1
                        )
                        mask_sliders &= (impr >= min_s) & (impr <= max_s)

            with col2:
                # Clicks
                if "Clics" in data_filtered.columns:
                    clicks = pd.to_numeric(
                        data_filtered["Clics"].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce"
                    )

                    if not clicks.dropna().empty:
                        min_v, max_v = int(clicks.min()), int(clicks.max())
                        min_s, max_s = st.slider(
                            "Clics",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v),
                            step=1
                        )
                        mask_sliders &= (clicks >= min_s) & (clicks <= max_s)

            with col3:
                # CTR
                if "CTR" in data_filtered.columns:
                    ctr = pd.to_numeric(
                        data_filtered["CTR"]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.replace(",", ".", regex=False),
                        errors="coerce"
                    )

                    if not ctr.dropna().empty:
                        min_v, max_v = float(ctr.min()), float(ctr.max())
                        min_s, max_s = st.slider(
                            "CTR (%)",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v)
                        )
                        mask_sliders &= (ctr >= min_s) & (ctr <= max_s)

            with col4:
                # CPC moyen
                if "CPC moy." in data_filtered.columns:
                    cpc = pd.to_numeric(
                        data_filtered["CPC moy."]
                        .astype(str)
                        .str.replace(",", ".", regex=False),
                        errors="coerce"
                    )

                    if not cpc.dropna().empty:
                        min_v, max_v = float(cpc.min()), float(cpc.max())
                        min_s, max_s = st.slider(
                            "CPC moyen",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v)
                        )
                        mask_sliders &= (cpc >= min_s) & (cpc <= max_s)

            with col5:
                # Coût
                if "Coût" in data_filtered.columns:
                    if not data_filtered["Coût"].empty:
                        min_v, max_v = float(data_filtered["Coût"].min()), float(data_filtered["Coût"].max())
                        min_s, max_s = st.slider(
                            "Coût",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v)
                        )
                        mask_sliders &= (data_filtered["Coût"] >= min_s) & (data_filtered["Coût"] <= max_s)

            with col6:
                # Taux de conversion target
                if "taux conv target" in data_filtered.columns:
                    conv = pd.to_numeric(
                        data_filtered["taux conv target"]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.replace(",", ".", regex=False),
                        errors="coerce"
                    )

                    if not conv.dropna().empty:
                        min_v, max_v = float(conv.min()), (float(conv.max()) if conv.max() != conv.min() else float(conv.max()) + 1)
                        min_s, max_s = st.slider(
                            "Taux conv target (%)",
                            min_value=min_v,
                            max_value=max_v,
                            value=(min_v, max_v)
                        )
                        mask_sliders &= (conv >= min_s) & (conv <= max_s)
                else:
                    st.info("Choose the target conv col")
            
            # Appliquer le masque des sliders
            data_selected = data_filtered.loc[mask_sliders].copy()

            # Sauvegarder dans session_state
            st.session_state["data_selected"] = data_selected
            
            # Afficher
            st.subheader(f"📊 Données filtrées ({len(data_selected)} lignes)")
            st.data_editor(data_selected, key="main_data_editor")

        else:
            #st.data_editor(data)
            st.warning("You need to choose a excel file")

    def general_graph(self):
        """Affiche un graphique combiné scatter + bar"""
        
        # Vérifier que les données existent
        if "data_selected" not in st.session_state:
            st.warning("⚠️ Aucune donnée sélectionnée. Retournez à la page principale.")
            return
        
        data_selected = st.session_state["data_selected"]
        
        if data_selected.empty:
            st.warning("⚠️ Aucune donnée à afficher")
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
            st.header("Parameter Model")
            list_model = [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
            
            with st.expander("Choose Model Parameter"):
                model_name = st.selectbox("Choose the model", list_model, key="sbert_model_name")
                auth_n_cluster = st.checkbox("Active KMeans n_cluster automatisation")
                active_cluster = st.button("Active clustering")
        
        data_cluster = st.session_state.get("data_selected")
        
        # ✅ Vérification que data_selected existe
        if data_cluster is None or data_cluster.empty:
            st.warning("⚠️ Aucune donnée sélectionnée pour le clustering")
            return
        
        standart_cluster_col = "Terme de recherche"
        texts = None

        if standart_cluster_col in data_cluster.columns:
            st.success(f"The cluster col is found: {standart_cluster_col}")
            texts = (
                data_cluster[standart_cluster_col]
                .fillna("")
                .str.strip()
            )
        else:
            col_cluster = st.text_input("Name of the cluster column")
            if col_cluster:
                texts = (
                    data_cluster[col_cluster]
                    .fillna("")
                    .str.strip()
                )
            else:
                st.warning("Choose the cluster column")

        if active_cluster and texts is not None:
            with st.spinner("Clustering en cours..."):
                emb = self._embed_texts(texts=tuple(texts), model_name=model_name)
                k_vals = 5
                best_k = None
                best_s = None

                if auth_n_cluster:
                    best_k, best_s = None, -1
                    for k in range(2, 20):
                        labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
                        s = silhouette_score(emb, labels=labels, metric="cosine")
                    
                        if s > best_s:
                            best_s, best_k = s, k

                    st.caption(f"the best k ist {best_k}\nThe score ist {best_s}")
                    labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(emb)

                else:
                    # ✅ Correction: définir best_k et best_s aussi pour le cas non-automatique
                    best_k = k_vals
                    labels = KMeans(n_clusters=k_vals, random_state=42).fit_predict(emb)
                    best_s = silhouette_score(emb, labels=labels, metric="cosine")
                
                data_cluster["cluster"] = labels

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best K", best_k)
                with col2:
                    st.metric("The silhouette", f"{best_s:.3f}")

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
                #"Coût/conv.": "mean"
            }
            

            if "taux conv target" in data_cluster.columns:
                agg_dict["taux conv target"] = "mean"

            group_by_cluster = data_cluster.groupby(["kw","cluster"]).agg(agg_dict).reset_index()
            
            numeric_cols = [col for col in agg_dict.keys()]
            
            scaled = group_by_cluster.copy()
            scaler = MinMaxScaler(feature_range=(0.1, 0.99))
            scaled[numeric_cols] = scaler.fit_transform(group_by_cluster[numeric_cols])

            scaled_melted = pd.melt(
                scaled, 
                id_vars=["kw","cluster"], 
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
            st.warning("⚠️ Lance d'abord le clustering + le cluster_barplot.")
            return

        df_cluster = st.session_state["scaled"].copy()      # agrégé par cluster + scaled
        data_cluster = st.session_state["df_cluster"].copy()  # lignes originales + cluster

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
        unique_cluster = df_cluster["cluster"].unique()
        selected_cluster = st.multiselect("select cluster", unique_cluster)
        if selected_cluster:
            data_cluster = data_cluster[data_cluster["cluster"].isin(selected_cluster)]
            df_cluster = df_cluster[df_cluster["cluster"].isin(selected_cluster)]
        
        group_kw = (
                data_cluster
                .groupby(["kw", "cluster"])["Terme de recherche"]
                .nunique()
                .reset_index(name="nb_search_terme")
            )
        col1, col2 = st.columns(2)
        row_h = 28
        chart_h = max(250, min(900, row_h * len(group_kw)))

        with col1:
            st.bar_chart(df_cluster, x="cluster", y=score_cols, stack=False, height=chart_h)
        with col2:
            st.bar_chart(group_kw, x="kw", y="nb_search_terme", color="cluster",
                        stack=False, horizontal=True, height=chart_h)




        # --- Tabs: 1 tab = 1 cluster ---
        tab_labels = [
            f"Cluster {int(row['cluster'])} | Opp {row['perf_opportunite_visibilite']:.2f}"
            for _, row in df_cluster.iterrows()
        ]
        tabs = st.tabs(tab_labels)

        for tab, (_, row) in zip(tabs, df_cluster.iterrows()):
            cluster_id = row["cluster"]
            kw_id = row["kw"]

            with tab:
                st.subheader(f"Cluster {cluster_id}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Opportunité visibilité", float(row["perf_opportunite_visibilite"]))
                col2.metric("Top performance", float(row.get("perf_top_performance", 0)))
                col3.metric("Risque", float(row.get("perf_risque", 0)))
                
                df_lines = pd.DataFrame(data_cluster[(data_cluster["cluster"] == cluster_id) & (data_cluster["kw"] == kw_id)].copy())
                st.caption("📌 Agrégé (scaled)")
                st.dataframe(row.to_frame().T, use_container_width=True)

                st.caption("🔎 Lignes du cluster (données brutes filtrées)")
                st.dataframe(df_lines, use_container_width=True)
        
        st.session_state["data_perf"] = df_cluster
    


            
    
if __name__ == "__main__":
    dashboard = StreamlitGADS()
    dashboard.main_dash()
    st.divider()
    dashboard.general_graph()
    dashboard.cluster()
    dashboard.cluster_barplot()
    dashboard.perf_barplot()
