import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import load_sample_data  # Assume this loads the 5 most recent seasons

def player_matchup_analyzer_page(model):
    st.markdown('<div class="section-header">Player Matchup Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Analyze individual player-vs-player matchups (e.g., WR vs CB, OL vs DL) using the 5 most recent NFL seasons.")

    # Load data (should be 5 most recent seasons)
    df = load_sample_data(full_data=True)
    if df is None:
        st.error("**Data Unavailable** - Unable to load player matchup data.")
        return
    # Fallback: If player columns are missing, try to create them from existing columns
    if 'posteam_player' not in df.columns:
        if 'receiver_player_name' in df.columns:
            df['posteam_player'] = df['receiver_player_name']
        elif 'rusher_player_name' in df.columns:
            df['posteam_player'] = df['rusher_player_name']
        else:
            df['posteam_player'] = df['posteam']
    if 'defender' not in df.columns:
        if 'defender_player_name' in df.columns:
            df['defender'] = df['defender_player_name']
        elif 'tackler_player_name' in df.columns:
            df['defender'] = df['tackler_player_name']
        else:
            df['defender'] = df['defteam']

    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        offense_player = st.selectbox("Select Offensive Player (e.g., WR, OL)", sorted(df['posteam_player'].dropna().unique()), help="Choose an offensive player to analyze")
    with col2:
        defense_player = st.selectbox("Select Defensive Player (e.g., CB, DL)", sorted(df['defender'].dropna().unique()), help="Choose a defensive player to analyze")

    # Situation filters
    st.markdown("<div class='subsection-header'>Situation Filters</div>", unsafe_allow_html=True)
    down = st.selectbox("Down", [1, 2, 3, 4])
    ydstogo = st.slider("Yards to Go", 1, 25, 10)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    play_type = st.selectbox("Play Type", ['run', 'pass'])

    # Filter data for matchup and situation
    matchup_df = df[
        (df['posteam_player'] == offense_player) &
        (df['defender'] == defense_player) &
        (df['down'] == down) &
        (df['ydstogo'] == ydstogo) &
        (df['quarter'] == quarter) &
        (df['play_type'] == play_type)
    ]

    st.markdown(f"<div class='info-card'><strong>Plays Found:</strong> {len(matchup_df)}</div>", unsafe_allow_html=True)

    if len(matchup_df) == 0:
        st.warning("No plays found for this matchup and situation in the last 5 seasons.")
        return

    # Show summary stats
    avg_yards = matchup_df['yards_gained'].mean()
    win_rate = (matchup_df['yards_gained'] >= matchup_df['ydstogo']).mean()
    st.metric("Average Yards Gained", f"{avg_yards:.2f}")
    st.metric("Success Rate (1st Down/TD)", f"{win_rate:.1%}")

    # Visualize yards gained distribution
    fig = px.histogram(matchup_df, x='yards_gained', nbins=20, title="Yards Gained Distribution in Matchup")
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data option
    if st.checkbox("Show Raw Matchup Data"):
        st.dataframe(matchup_df, use_container_width=True) 