import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import load_sample_data

def player_matchup_analyzer_page(model):
    st.markdown('<div class="section-header">Player Performance Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Analyze individual player performance across different game situations and matchups")

    # Load data
    df = load_sample_data(full_data=True)
    if df is None:
        st.error("**Data Unavailable** - Unable to load player data.")
        return

    # Build a unified player list from both receiver and rusher columns
    player_cols = []
    if 'receiver_player_name' in df.columns:
        player_cols.append('receiver_player_name')
    if 'rusher_player_name' in df.columns:
        player_cols.append('rusher_player_name')
    if not player_cols:
        st.error("No player columns found in data.")
        return

    # Get all unique player names (excluding NaN)
    player_names = set()
    for col in player_cols:
        player_names.update(df[col].dropna().unique())
    player_names = sorted(player_names)

    # Search bar for player selection
    search_query = st.text_input(
        "Search for Player to Analyze",
        "",
        help="Type a player's name to search for their stats"
    )
    filtered_players = [p for p in player_names if search_query.lower() in p.lower()]
    if not filtered_players and search_query:
        st.warning(f"No players found matching '{search_query}'")
        return
    selected_player = st.selectbox(
        "Select Player from Results",
        filtered_players if filtered_players else player_names,
        help="Select a player from the search results"
    )

    # Season selection (only if season data is available)
    if 'season' in df.columns:
        available_seasons = sorted(df['season'].unique())
        selected_season = st.selectbox(
            "Select Season",
            ["All Seasons"] + list(available_seasons),
            help="Choose a specific season or view all seasons"
        )
    else:
        selected_season = "All Seasons"

    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Performance by Down", "Performance by Quarter", "Performance by Field Position", "Performance vs Teams"] + 
        (["Seasonal Performance"] if 'season' in df.columns else []),
        help="Choose what aspect of performance to analyze"
    )

    # Filter data for selected player and season
    # For pass plays, match receiver_player_name; for run plays, match rusher_player_name
    player_df = pd.DataFrame()
    if 'receiver_player_name' in df.columns:
        player_df = pd.concat([
            player_df,
            df[(df['play_type'] == 'pass') & (df['receiver_player_name'] == selected_player)]
        ])
    if 'rusher_player_name' in df.columns:
        player_df = pd.concat([
            player_df,
            df[(df['play_type'] == 'run') & (df['rusher_player_name'] == selected_player)]
        ])
    if selected_season != "All Seasons" and 'season' in player_df.columns:
        player_df = player_df[player_df['season'] == selected_season]

    if len(player_df) == 0:
        st.warning(f"No data found for {selected_player}" + 
                  (f" in {selected_season}" if selected_season != "All Seasons" else ""))
        return

    # Display overall stats
    st.markdown('<div class="subsection-header">Overall Performance</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_yards = player_df['yards_gained'].mean()
        st.metric("Average Yards per Play", f"{avg_yards:.2f}")
    with col2:
        success_rate = (player_df['yards_gained'] >= player_df['ydstogo']).mean()
        st.metric("Success Rate", f"{success_rate:.1%}")
    with col3:
        total_plays = len(player_df)
        st.metric("Total Plays", f"{total_plays:,}")
    with col4:
        pass_pct = (player_df['play_type'] == 'pass').mean()
        st.metric("Pass Play %", f"{pass_pct:.1%}")

    # Analysis based on selected type
    st.markdown(f'<div class="subsection-header">{analysis_type}</div>', unsafe_allow_html=True)
    
    if analysis_type == "Seasonal Performance" and 'season' in df.columns:
        # Group by season and calculate metrics
        season_stats = player_df.groupby('season').agg({
            'yards_gained': ['mean', 'count'],
            'play_type': lambda x: (x == 'pass').mean()
        }).round(2)
        
        season_stats.columns = ['Average Yards', 'Play Count', 'Pass %']
        season_stats = season_stats.reset_index()
        
        # Create visualization
        fig = px.bar(
            season_stats,
            x='season',
            y='Average Yards',
            color='Average Yards',
            title=f"{selected_player}'s Performance by Season",
            text='Average Yards',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats
        st.dataframe(season_stats, use_container_width=True)
        
        # Add trend analysis
        if len(season_stats) > 1:
            trend = season_stats['Average Yards'].pct_change().mean() * 100
            trend_color = "green" if trend > 0 else "red"
            st.markdown(f"""
            <div class="info-card">
                <strong>Performance Trend:</strong> 
                <span style="color: {trend_color}">
                    {trend:+.1f}% average change per season
                </span>
            </div>
            """, unsafe_allow_html=True)

    elif analysis_type == "Performance by Down":
        # Group by down and calculate metrics
        down_stats = player_df.groupby('down').agg({
            'yards_gained': ['mean', 'count'],
            'play_type': lambda x: (x == 'pass').mean()
        }).round(2)
        
        down_stats.columns = ['Average Yards', 'Play Count', 'Pass %']
        down_stats = down_stats.reset_index()
        
        # Create visualization
        fig = px.bar(
            down_stats,
            x='down',
            y='Average Yards',
            color='Average Yards',
            title=f"{selected_player}'s Performance by Down",
            text='Average Yards',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats
        st.dataframe(down_stats, use_container_width=True)

    elif analysis_type == "Performance by Quarter":
        # Group by quarter and calculate metrics
        quarter_stats = player_df.groupby('quarter').agg({
            'yards_gained': ['mean', 'count'],
            'play_type': lambda x: (x == 'pass').mean()
        }).round(2)
        
        quarter_stats.columns = ['Average Yards', 'Play Count', 'Pass %']
        quarter_stats = quarter_stats.reset_index()
        
        # Create visualization
        fig = px.bar(
            quarter_stats,
            x='quarter',
            y='Average Yards',
            color='Average Yards',
            title=f"{selected_player}'s Performance by Quarter",
            text='Average Yards',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats
        st.dataframe(quarter_stats, use_container_width=True)

    elif analysis_type == "Performance by Field Position":
        # Create field position zones
        player_df['field_zone'] = pd.cut(
            player_df['yardline_100'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Goal Line (1-20)', 'Red Zone (21-40)', 'Short Field (41-60)', 'Mid Field (61-80)', 'Long Field (81+)']
        )
        
        # Group by field zone and calculate metrics
        zone_stats = player_df.groupby('field_zone').agg({
            'yards_gained': ['mean', 'count'],
            'play_type': lambda x: (x == 'pass').mean()
        }).round(2)
        
        zone_stats.columns = ['Average Yards', 'Play Count', 'Pass %']
        zone_stats = zone_stats.reset_index()
        
        # Create visualization
        fig = px.bar(
            zone_stats,
            x='field_zone',
            y='Average Yards',
            color='Average Yards',
            title=f"{selected_player}'s Performance by Field Position",
            text='Average Yards',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats
        st.dataframe(zone_stats, use_container_width=True)

    elif analysis_type == "Performance vs Teams":
        # Group by defensive team and calculate metrics
        team_stats = player_df.groupby('defteam').agg({
            'yards_gained': ['mean', 'count'],
            'play_type': lambda x: (x == 'pass').mean()
        }).round(2)
        
        team_stats.columns = ['Average Yards', 'Play Count', 'Pass %']
        team_stats = team_stats.reset_index()
        
        # Create visualization
        fig = px.bar(
            team_stats,
            x='defteam',
            y='Average Yards',
            color='Average Yards',
            title=f"{selected_player}'s Performance vs Teams",
            text='Average Yards',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed stats
        st.dataframe(team_stats, use_container_width=True)

    # Show raw data option
    if st.checkbox("Show Raw Player Data"):
        st.dataframe(player_df, use_container_width=True) 