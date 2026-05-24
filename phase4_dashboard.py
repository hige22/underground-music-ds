"""
Phase 4 — Streamlit Dashboard
Project: What Does Underground Sound Like?

Run this with:
    streamlit run phase4_dashboard.py

Make sure all Phase 1–3 files are in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="What Does Underground Sound Like?",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD DATA AND MODELS
# We use @st.cache_data so data loads only ONCE.
# Without this, it reloads on every interaction.
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('df_with_clusters.csv')
    df_clean = pd.read_csv('df_clean.csv')
    profiles = pd.read_csv('cluster_profiles.csv', index_col='cluster')
    return df, df_clean, profiles

@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca    = joblib.load('pca_model.pkl')
    return kmeans, scaler, pca

df, df_clean, cluster_profiles = load_data()
kmeans, scaler, pca = load_models()

FEATURES = ['danceability', 'energy', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence']

C_UNDERGROUND = '#5B4CF0'
C_MAINSTREAM  = '#E8593C'
CLUSTER_COLORS = ['#5B4CF0', '#E8593C', '#1DAE77', '#E8A020']

# Cluster names — edit these based on your Cell 7 findings!
CLUSTER_NAMES = {
    0: 'Atmospheric & Introspective',
    1: 'High Energy & Dance',
    2: 'Acoustic & Raw',
    3: 'Polished Pop',
    4: 'Experimental Underground',
    5: 'Rhythmic & Groovy'
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Underground Music Explorer")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔬 EDA Deep Dive", "🤖 Cluster Explorer", "🔍 Song Finder"]
    )

    st.markdown("---")
    st.caption("Data: 30,000 Spotify songs")
    st.caption("Model: K-Means (K=4) + PCA")
    st.caption("Built with Python + Streamlit")

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":

    st.title("What Does Underground Sound Like?")
    st.markdown(
        "A data science exploration of how underground and mainstream music differ "
        "across Spotify's audio features — using clustering and PCA to reveal hidden sonic patterns."
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    total_songs = len(df)
    n_underground = len(df[df['label'] == 'underground'])
    n_mainstream  = len(df[df['label'] == 'mainstream'])
    n_clusters    = df['cluster'].nunique()

    with col1:
        st.metric("Total Songs Analysed", f"{total_songs:,}")
    with col2:
        st.metric("Underground Songs", f"{n_underground:,}")
    with col3:
        st.metric("Mainstream Songs", f"{n_mainstream:,}")
    with col4:
        st.metric("Sonic Clusters Found", n_clusters)

    st.markdown("---")

    # Popularity distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Popularity Distribution")

        fig = px.histogram(
            df_clean, x='track_popularity',
            color='label',
            color_discrete_map={
                'underground': C_UNDERGROUND,
                'mainstream': C_MAINSTREAM,
                'mid': '#cccccc'
            },
            nbins=50,
            labels={'track_popularity': 'Popularity Score', 'count': 'Songs'},
            title='How songs are distributed by popularity score'
        )
        fig.update_layout(bargap=0.05, legend_title='Label')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Radar: Sound DNA Comparison")

        ug_means  = df[df['label'] == 'underground'][FEATURES].mean()
        ms_means  = df[df['label'] == 'mainstream'][FEATURES].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=ug_means.values.tolist() + [ug_means.values[0]],
            theta=FEATURES + [FEATURES[0]],
            fill='toself', name='Underground',
            line_color=C_UNDERGROUND, fillcolor=C_UNDERGROUND,
            opacity=0.5
        ))
        fig.add_trace(go.Scatterpolar(
            r=ms_means.values.tolist() + [ms_means.values[0]],
            theta=FEATURES + [FEATURES[0]],
            fill='toself', name='Mainstream',
            line_color=C_MAINSTREAM, fillcolor=C_MAINSTREAM,
            opacity=0.5
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Average sound profile comparison'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key insight callout
    st.info(
        "🔍 **Key Finding:** Underground songs score notably higher on **acousticness** and "
        "**instrumentalness**, while mainstream songs lead on **danceability** and **energy**. "
        "This suggests underground music tends toward rawer, more complex sonic textures."
    )

# ─────────────────────────────────────────────
# PAGE 2 — EDA DEEP DIVE
# ─────────────────────────────────────────────
elif page == "🔬 EDA Deep Dive":

    st.title("🔬 EDA Deep Dive")
    st.markdown("Explore how each audio feature differs between underground and mainstream music.")

    # Feature selector
    selected_feature = st.selectbox(
        "Select an audio feature to explore:",
        FEATURES,
        format_func=lambda x: x.capitalize()
    )

    col1, col2 = st.columns(2)

    with col1:
        # Violin plot for selected feature
        plot_data = pd.concat([
            df[df['label'] == 'underground'][[selected_feature]].assign(Label='Underground'),
            df[df['label'] == 'mainstream'][[selected_feature]].assign(Label='Mainstream')
        ])

        fig = px.violin(
            plot_data, x='Label', y=selected_feature,
            color='Label',
            color_discrete_map={'Underground': C_UNDERGROUND, 'Mainstream': C_MAINSTREAM},
            box=True, points=False,
            title=f'{selected_feature.capitalize()} distribution by label'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Summary stats comparison
        ug_stats = df[df['label'] == 'underground'][selected_feature].describe()
        ms_stats = df[df['label'] == 'mainstream'][selected_feature].describe()

        stats_df = pd.DataFrame({
            'Underground': ug_stats,
            'Mainstream': ms_stats
        }).round(3)

        st.markdown(f"**{selected_feature.capitalize()} — Summary Statistics**")
        st.dataframe(stats_df, use_container_width=True)

        diff = df[df['label'] == 'underground'][selected_feature].mean() - \
               df[df['label'] == 'mainstream'][selected_feature].mean()

        if diff > 0.02:
            st.success(f"Underground songs have **higher** {selected_feature} (+{diff:.3f})")
        elif diff < -0.02:
            st.warning(f"Underground songs have **lower** {selected_feature} ({diff:.3f})")
        else:
            st.info(f"Both groups have **similar** {selected_feature} (diff: {diff:.3f})")

    # Genre breakdown
    st.markdown("---")
    st.subheader("Genre Breakdown")

    genre_data = df.groupby(['playlist_genre', 'label']).size().reset_index(name='count')
    genre_total = df.groupby('playlist_genre').size().reset_index(name='total')
    genre_pct = genre_data[genre_data['label'] == 'underground'].merge(genre_total, on='playlist_genre')
    genre_pct['underground_pct'] = (genre_pct['count'] / genre_pct['total'] * 100).round(1)
    genre_pct = genre_pct.sort_values('underground_pct', ascending=True)

    fig = px.bar(
        genre_pct, x='underground_pct', y='playlist_genre',
        orientation='h',
        color='underground_pct',
        color_continuous_scale=[[0, C_MAINSTREAM], [1, C_UNDERGROUND]],
        labels={'underground_pct': '% Underground', 'playlist_genre': 'Genre'},
        title='What % of each genre is "underground"?'
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — CLUSTER EXPLORER
# ─────────────────────────────────────────────
elif page == "🤖 Cluster Explorer":

    st.title("🤖 Cluster Explorer")
    st.markdown(
        "K-Means found **4 distinct sonic profiles** in the data — "
        "purely from the audio features, without knowing popularity."
    )

    # PCA scatter plot
    st.subheader("Songs in 2D Audio Feature Space")

    color_by = st.radio("Colour points by:", ["Cluster", "Label"], horizontal=True)

    if color_by == "Cluster":
        df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)
        fig = px.scatter(
            df.sample(min(5000, len(df)), random_state=42),  # sample for speed
            x='pc1', y='pc2',
            color='cluster_name',
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.5, size_max=6,
            hover_data=['track_name', 'track_artist', 'track_popularity'],
            labels={'pc1': 'PC1 (audio component 1)', 'pc2': 'PC2 (audio component 2)'},
            title='Hover over any point to see the song'
        )
    else:
        fig = px.scatter(
            df.sample(min(5000, len(df)), random_state=42),
            x='pc1', y='pc2',
            color='label',
            color_discrete_map={'underground': C_UNDERGROUND, 'mainstream': C_MAINSTREAM},
            opacity=0.5,
            hover_data=['track_name', 'track_artist', 'track_popularity'],
            title='Hover over any point to see the song'
        )

    st.plotly_chart(fig, use_container_width=True)

    # Cluster profiles
    st.markdown("---")
    st.subheader("Cluster Audio Profiles")

    cols = st.columns(4)
    for k, col in enumerate(cols):
        with col:
            cluster_name = CLUSTER_NAMES.get(k, f'Cluster {k}')
            st.markdown(f"**{cluster_name}**")

            profile = cluster_profiles.loc[k]
            radar_features = [f for f in FEATURES if f in profile.index]

            fig = go.Figure(go.Scatterpolar(
                r=[profile[f] for f in radar_features] + [profile[radar_features[0]]],
                theta=radar_features + [radar_features[0]],
                fill='toself',
                line_color=CLUSTER_COLORS[k],
                fillcolor=CLUSTER_COLORS[k],
                opacity=0.6
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            n_songs = len(df[df['cluster'] == k])
            ug_pct  = len(df[(df['cluster'] == k) & (df['label'] == 'underground')]) / n_songs * 100
            st.caption(f"{n_songs:,} songs · {ug_pct:.0f}% underground")

# ─────────────────────────────────────────────
# PAGE 4 — SONG FINDER
# ─────────────────────────────────────────────
elif page == "🔍 Song Finder":

    st.title("🔍 Song Finder")
    st.markdown("Search for any song in the dataset and see its audio fingerprint and cluster.")

    # Search bar
    search_query = st.text_input("Search by song name or artist:", placeholder="e.g. Radiohead, Dua Lipa...")

    if search_query:
        # Filter songs matching the query
        mask = (
            df['track_name'].str.lower().str.contains(search_query.lower(), na=False) |
            df['track_artist'].str.lower().str.contains(search_query.lower(), na=False)
        )
        results = df[mask][['track_name', 'track_artist', 'track_popularity', 'label', 'cluster']].head(20)

        if len(results) == 0:
            st.warning(f"No songs found matching '{search_query}'")
        else:
            st.success(f"Found {len(results)} matching songs:")
            results['Cluster Name'] = results['cluster'].map(CLUSTER_NAMES)
            st.dataframe(
                results.rename(columns={
                    'track_name': 'Song',
                    'track_artist': 'Artist',
                    'track_popularity': 'Popularity',
                    'label': 'Label',
                    'cluster': 'Cluster ID'
                }),
                use_container_width=True
            )

            # Show audio features for the first result
            first = df[mask].iloc[0]
            st.markdown(f"---\n**Audio fingerprint of: {first['track_name']} — {first['track_artist']}**")

            fig = go.Figure(go.Scatterpolar(
                r=[first[f] for f in FEATURES] + [first[FEATURES[0]]],
                theta=FEATURES + [FEATURES[0]],
                fill='toself',
                line_color=C_UNDERGROUND if first['label'] == 'underground' else C_MAINSTREAM,
                opacity=0.7
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Label: **{first['label'].upper()}** · "
                f"Cluster: **{CLUSTER_NAMES.get(first['cluster'], first['cluster'])}** · "
                f"Popularity: **{int(first['track_popularity'])}**"
            )
