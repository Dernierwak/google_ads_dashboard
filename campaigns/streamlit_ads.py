# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import altair as alt
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# =========================
# Config App
# =========================
# Éviter de reconfigurer plusieurs fois
if "_page_config_set" not in st.session_state:
    st.set_page_config(page_title="Meta Dashboard", layout="wide")
    st.session_state["_page_config_set"] = True

# Import user-agent (optionnel, commenté si non disponible)
try:
    from ai.agent import invok_agent
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False
    st.warning("Module 'user_agent' non trouvé. La fonctionnalité d'analyse IA est désactivée.")


# =========================
# Data
# =========================
@st.cache_data(show_spinner=False)
def load_data(json_dir: str) -> pd.DataFrame:
    """Charge et fusionne les JSON en un seul DataFrame."""
    path_insights = os.path.join(json_dir, "ads_insights.json")
    path_assets = os.path.join(json_dir, "ads_assets.json")

    if not os.path.exists(path_insights):
        raise FileNotFoundError(f"Fichier introuvable: {path_insights}")
    if not os.path.exists(path_assets):
        raise FileNotFoundError(f"Fichier introuvable: {path_assets}")

    df_ins = pd.read_json(path_insights)
    df_ast = pd.read_json(path_assets)
    #st.markdown(path_assets)

    # types
    st.dataframe(df_ast)
    #st.dataframe(df_ins)

    if "ads_id" in df_ins.columns:
        df_ins["ads_id"] = df_ins["ads_id"].astype(str)
    if "ads_id" in df_ast.columns:
        df_ast["ads_id"] = df_ast["ads_id"].astype(str)

    df = pd.merge(df_ast, df_ins, how="right", on="ads_id")

    # colonnes minimales attendues (on les crée si manquantes pour éviter les plantages)
    for col, default in [
        ("campaigns_name", None),
        ("adset_name", None),
        ("ads_name_y", None),     # nom d'annonce (côté insights)
        ("impressions", 0),
        ("clicks", 0),
        ("ctr", 0.0),
        ("spend", 0.0),
        ("body", None),
        ("title", None),
        ("body_assets", None),
        ("title_assets", None),
    ]:
        if col not in df.columns:
            df[col] = default

    # normalisation types numériques
    for col in ["impressions", "clicks"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["ctr", "spend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # normalisation strings
    for col in ["campaigns_name", "adset_name", "ads_name_y", "body", "title"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"None": "", "nan": ""})

    # date_start si présente
    if "date_start" in df.columns:
        try:
            df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        except Exception:
            pass

    return df


# =========================
# UI Helpers
# =========================
def reset_adsets():
    """Reset les adsets sélectionnés quand les campagnes changent."""
    st.session_state["adset_choices"] = []


# =========================
# Filtres (sidebar)
# =========================
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les filtres de la sidebar et retourne le DataFrame filtré."""
    st.sidebar.header("Filtres")

    # 1) Campagnes
    campaigns = sorted([c for c in df["campaigns_name"].dropna().unique() if c])
    selected_campaigns = st.sidebar.multiselect(
        "Campagnes",
        options=campaigns,
        default=st.session_state.get("campaign_choices", []),
        key="campaign_choices",
        on_change=reset_adsets
    )

    df_f = df.copy()
    if selected_campaigns:
        df_f = df_f[df_f["campaigns_name"].isin(selected_campaigns)]

    # 2) Adsets (dépend des campagnes)
    adsets = sorted([a for a in df_f["adset_name"].dropna().unique() if a])
    selected_adsets = st.sidebar.multiselect(
        "Ad sets",
        options=adsets,
        default=st.session_state.get("adset_choices", []),
        key="adset_choices"
    )

    if selected_adsets:
        df_f = df_f[df_f["adset_name"].isin(selected_adsets)]

    # 3) Filtre texte sur adset
    text_value = st.sidebar.text_input(
        "Contient (dans Ad set)", 
        value=st.session_state.get("text_filter", ""),
        key="text_filter_input"
    )
    st.session_state["text_filter"] = text_value
    if text_value:
        df_f = df_f[df_f["adset_name"].str.contains(text_value, case=False, na=False)]

    # 4) Filtre texte sur ads_name
    text_ads = st.sidebar.text_input(
        "Contient (dans Nom d'annonce)", 
        value=st.session_state.get("text_filter_ads", ""),
        key="text_filter_ads_input"
    )
    st.session_state["text_filter_ads"] = text_ads
    if text_ads:
        df_f = df_f[df_f["ads_name_y"].str.contains(text_ads, case=False, na=False)]

    return df_f


# =========================
# Sliders + Scatter(s) + Tiers + Propagation
# =========================
def chart_and_sliders(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Construit l'agrégat, affiche sliders + scatters + tiers,
       et renvoie le DF de base filtré (avec 'tier' propagé)."""
    st.subheader("📊 Performance Annonces")

    if df_filtered.empty:
        st.info("Aucune donnée selon les filtres sélectionnés.")
        return df_filtered.iloc[0:0].copy()
    
    # agrégation par (adset, ad)
    metrics_agg = {
            "impressions": "sum",
            "clicks": "sum",
            "ctr": "mean",
            "spend": "sum",
            "link_click": "sum"
        }
    if "metrics_agg" not in st.session_state:
        st.session_state["metrics_agg"] = metrics_agg

    grouped = (
        df_filtered.groupby(["adset_name", "ads_name_y"], as_index=False, dropna=False)
        .agg(metrics_agg)
    )
    #st.header("ici")
    #st.data_editor(grouped)
    if grouped.empty:
        st.info("Aucune donnée agrégée disponible.")
        return df_filtered.iloc[0:0].copy()

    # bornes pour sliders
    min_impr = int(grouped["impressions"].min())
    max_impr = int(grouped["impressions"].max())
    min_click = int(grouped["clicks"].min())
    max_click = int(grouped["clicks"].max())
    min_cost = float(grouped["spend"].min())
    max_cost = float(grouped["spend"].max())

    # sécurités sliders (éviter min == max)
    if min_impr == max_impr:
        max_impr = min_impr + 1
    if min_click == max_click:
        max_click = min_click + 1
    if min_cost == max_cost:
        max_cost = min_cost + 1.0

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        impr_range = st.slider(
            "Impressions",
            min_value=min_impr,
            max_value=max_impr,
            value=(min_impr, max_impr),
            key="slider_impr"
        )
    with c2:
        click_range = st.slider(
            "Clicks",
            min_value=min_click,
            max_value=max_click,
            value=(min_click, max_click),
            key="slider_clicks"
        )
    with c3:
        cost_range = st.slider(
            "Spend ($)",
            min_value=min_cost,
            max_value=max_cost,
            value=(min_cost, max_cost),
            key="slider_spend"
        )

    # application sliders sur l'agrégat
    df_group_f = grouped[
        grouped["impressions"].between(impr_range[0], impr_range[1])
        & grouped["clicks"].between(click_range[0], click_range[1])
        & grouped["spend"].between(cost_range[0], cost_range[1])
    ].copy()

    if df_group_f.empty:
        st.warning("Aucune donnée après application des sliders.")
        return df_filtered.iloc[0:0].copy()

    # ---------- Scatter simple ----------
    st.markdown("### 🔵 Scatter simple")
    try:
        chart_basic = (
            alt.Chart(df_group_f)
            .mark_circle()
            .encode(
                x=alt.X("clicks:Q", title="Clicks"),
                y=alt.Y("impressions:Q", title="Impressions"),
                color=alt.Color("ads_name_y:N", title="Annonce", legend=None),
                size=alt.Size("spend:Q", title="Spend"),
                tooltip=[
                    alt.Tooltip("adset_name:N", title="Ad set"),
                    alt.Tooltip("ads_name_y:N", title="Annonce"),
                    alt.Tooltip("impressions:Q"),
                    alt.Tooltip("clicks:Q"),
                    alt.Tooltip("ctr:Q", format=".2%"),
                    alt.Tooltip("spend:Q", format=",.2f"),
                ],
            )
            .interactive()
            .properties(height=380)
        )
        st.altair_chart(chart_basic, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la création du scatter simple : {e}")

    # ========= Notation / tier =========
    st.markdown("### 🏷️ Classification par tiers")
    
    #metrics = ["ctr", "clicks", "impressions", "spend"]
    metrics = metrics_agg.keys()
    metric = st.selectbox("Métrique de ranking (tiers)", options=metrics, index=0, key="tier_metric")

    # Calcul des tiers
    s = pd.to_numeric(df_group_f[metric], errors="coerce").fillna(0)
    
    if s.min() == s.max() or len(s.unique()) < 3:
        df_group_f["tier"] = "Moyen"
        st.caption("⚠️ Métrique constante ou trop peu de valeurs uniques → tous les éléments sont classés 'Moyen'")
    else:
        try:
            df_group_f["tier"] = pd.qcut(s, q=3, labels=["Bas", "Moyen", "Haut"], duplicates='drop')
        except ValueError:
            df_group_f["tier"] = "Moyen"
            st.caption("⚠️ Impossible de créer 3 tiers distincts → classification par défaut")

    # Mettre 'tier' en première colonne
    cols = ["tier"] + [c for c in df_group_f.columns if c != "tier"]
    df_group_f = df_group_f[cols]

    # ---------- Scatter avancé : tiers + tendances ----------
    st.markdown("### 🔵 Scatter avancé — Tiers + tendances")
    try:
        scatter = (
            alt.Chart(df_group_f)
            .mark_circle()
            .encode(
                x=alt.X("clicks:Q", title="Clicks"),
                y=alt.Y("impressions:Q", title="Impressions"),
                color=alt.Color("tier:N", title="Tier", 
                    scale=alt.Scale(domain=["Bas", "Moyen", "Haut"], 
                                   range=["#e74c3c", "#f39c12", "#2ecc71"])),
                size=alt.Size("spend:Q", title="Spend", scale=alt.Scale(range=[50, 800])),
                tooltip=[
                    alt.Tooltip("tier:N"),
                    alt.Tooltip("adset_name:N", title="Ad set"),
                    alt.Tooltip("ads_name_y:N", title="Annonce"),
                    alt.Tooltip("impressions:Q", format=","),
                    alt.Tooltip("clicks:Q", format=","),
                    alt.Tooltip("ctr:Q", format=".2%"),
                    alt.Tooltip("spend:Q", format=",.2f"),
                ],
            )
        )

        # Ligne de régression linéaire
        reg_line = (
            alt.Chart(df_group_f)
            .transform_regression("clicks", "impressions")
            .mark_line(size=3, color="black")
        )

        # Ligne LOESS (tendance locale)
        loess_line = (
            alt.Chart(df_group_f)
            .transform_loess("clicks", "impressions", bandwidth=0.5)
            .mark_line(strokeDash=[4, 3], color="gray", opacity=0.7)
        )

        st.altair_chart(
            alt.layer(scatter, reg_line, loess_line).interactive().properties(height=420),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Erreur lors de la création du scatter avancé : {e}")

    # ---------- Tableau agrégé ----------
    st.markdown("#### 📋 Détail agrégé (après sliders)")
    df_group_f["link_click_ctr"] = round(df_group_f["link_click"] * 100 / df_group_f["impressions"], 2)
    st.dataframe(df_group_f, use_container_width=True)

    # Statistiques par tier
    #if "tier" in df_group_f.columns:
    #    st.markdown("#### 📈 Statistiques par tier")
    #    tier_stats = df_group_f.groupby("ads_name_y").agg({
    #        "impressions": ["sum", "mean"],
    #        "clicks": ["sum", "mean"],
    #       "spend": ["sum", "mean"],
    #       "ctr": "mean",
    #       "link_click": ["sum", "mean"]
    #   }).round(2)
    #   st.dataframe(tier_stats, use_container_width=True)

    # >>> Propagation des sliders + tier vers le DF de base
    df_after_sliders = df_filtered.merge(
        df_group_f[["adset_name", "ads_name_y", "tier"]].drop_duplicates(),
        on=["adset_name", "ads_name_y"],
        how="inner"
    )
    
    return df_after_sliders


# =========================
# Assets
# =========================
def render_assets(df_assets: pd.DataFrame):
    """Affiche la section Assets avec les créatifs."""
    st.subheader("🎨 Assets & Créatifs")
    

    if df_assets.empty:
        st.info("Aucun asset à afficher après application des filtres.")
        return


    # Colonnes à afficher
    cols = [c for c in [
        "tier",
        "campaigns_name", "adset_name", "ads_name_y",
        "body", "title", "body_assets", "title_assets",
        "impressions", "clicks", "ctr", "spend", "preview_shareable_link_y"
    ] if c in df_assets.columns]

    # Version groupée pour lecture rapide des créas
    agg_dict = st.session_state["metrics_agg"]
    
    # Ajouter les colonnes textuelles
    for col in ["tier", "body", "title", "body_assets", "title_assets", "preview_shareable_link_y"]:
        if col in df_assets.columns:
            agg_dict[col] = "first"

    grouped_asset = (
        df_assets.groupby(["adset_name", "ads_name_y"], as_index=False, dropna=False)
        .agg(agg_dict).round(2)
    )

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        tier = grouped_asset["tier"].unique()
        filter_tier = st.multiselect("Choose the Class", tier)


    grouped_asset["link_click_ctr"] = round(grouped_asset["link_click"] * 100 / grouped_asset["impressions"], 2)

    if filter_tier:
        grouped_asset = grouped_asset[grouped_asset["tier"].isin(filter_tier)]

    st.data_editor(grouped_asset, use_container_width=True)

    with st.expander("View the assets", expanded=False):
        body_assets = grouped_asset["body_assets"]
        st.dataframe(body_assets, width="stretch")


    # Line chart si date_start existe
    if "date_start" in df_assets.columns and pd.api.types.is_datetime64_any_dtype(df_assets["date_start"]):
        st.markdown("#### 📅 Évolution temporelle des impressions")
        try:
            # Agrégation par date et annonce
            time_series = df_assets.groupby(["date_start", "ads_name_y"], as_index=False)["impressions"].sum()
            st.line_chart(time_series, x="date_start", y="impressions", color="ads_name_y")
        except Exception as e:
            st.caption(f"Impossible d'afficher la courbe : {e}")
    else:
        st.caption("ℹ️ Aucune colonne 'date_start' utilisable pour la courbe temporelle.")

    # Export CSV
    c1, c2 = st.columns([1, 3])
    with c1:
        csv = df_assets[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export CSV",
            csv, 
            file_name="assets_filtres.csv", 
            mime="text/csv"
        )


# =========================
# Analyse IA (optionnelle)
# =========================
def ai_analysis_section(df: pd.DataFrame):
    """Section d'analyse IA si le module user_agent est disponible."""
    if not HAS_AGENT:
        return
    
    st.subheader("🤖 Analyse IA")
    
    if df.empty:
        st.info("Aucune donnée à analyser.")
        return
    
    col1, col2 = st.columns([3,1])
    with col1:
        message_agent = st.text_area("Speak to Agent")
    
    with col2:
        st.markdown("Default: Analyse ces données marketing Facebook Ads et donne-moi des insights actionnables.")
    
    if st.button("Analyser les données avec l'IA", key="ai_analysis_btn"):
        with st.spinner("Analyse en cours..."):
            try:
                agent = invok_agent(df)

                if message_agent:
                    response = agent.invoke(message_agent)
                else:
                    response = agent.invoke("Analyse ces données marketing Facebook Ads et donne-moi des insights actionnables.")
                st.markdown("### 📊 Résultat de l'analyse")
                st.write(response.get("output"))
            except Exception as e:
                st.error(f"Erreur lors de l'analyse IA : {e}")


# =========================
# Main
# =========================
def main():
    st.title("📱 Meta Dashboard")

    # chemin des JSON
    try:
        from core.paths import ADS_JSON_DIR
        JSON_DIR = ADS_JSON_DIR
    except Exception:
        JSON_DIR = Path(__file__).parent / "data_facebook_ads" / "json"
    
    try:
        df = load_data(str(JSON_DIR))
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("Assure-toi que les fichiers 'ads_insights.json' et 'ads_assets.json' sont présents dans le dossier 'data_facebook_ads/json/'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        st.stop()

    # Affichage des métadonnées
    with st.expander("ℹ️ Informations sur les données", expanded=False):
        st.write(f"**Nombre total de lignes :** {len(df)}")
        st.write(f"**Colonnes disponibles :** {', '.join(df.columns.tolist())}")
        st.write(f"**Période :** {df['date_start'].min()} à {df['date_start'].max()}" if "date_start" in df.columns else "Date non disponible")
    
    st.data_editor(df)

    df = df.astype({
        "ads_id": "string",
        "ads_name_x": "string",
        "ads_name_y": "string",
        "adset_name": "string",
        "campaigns_name": "string",
        "preview_shareable_link_x": "string",
        "preview_shareable_link_y": "string",
        "image_url": "string",
        "call_to_action": "string",
        "object_type": "string",
        "omni_landing_page_view": "string"
    })
    num_cols = [
        "impressions",
        "clicks",
        "inline_link_clicks",
        "link_click",
        "spend",
        "ctr",
        "cpc",
        "cpm",
        "reach",
        "page_engagement",
        "post_engagement",
        "post_reaction"
    ]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)



    # Filtres majeurs (sidebar)
    df_head_filtered = sidebar_filters(df)

    # Sliders + scatter(s) + propagation des tiers
    df_final = chart_and_sliders(df_head_filtered)

    # Assets filtrés (après sliders)
    render_assets(df_final)

    # Analyse IA (optionnelle)
    #ai_analysis_section(df_head_filtered)


if __name__ == "__main__":
    main()
