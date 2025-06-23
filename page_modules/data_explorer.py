import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import load_data

def data_explorer_page():
    df = load_data()
    if df is None:
        st.error("**Data Unavailable** - Unable to load exploration data")
        return
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("Interactive exploration and analysis of NFL play-by-play data")
    
    # dataset overview
    st.markdown('<div class="subsection-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Plays", f"{len(df):,}")
    with col2:
        unique_teams = df['posteam'].nunique() if 'posteam' in df.columns else "Various"
        st.metric("Teams Represented", str(unique_teams))
    with col3:
        years_span = "2019-2021"
        st.metric("Season Coverage", years_span)
    with col4:
        avg_yards = df['yards_gained'].mean()
        st.metric("Average Yards per Play", f"{avg_yards:.2f}")
    
    # advanced data filtering interface
    st.markdown('<div class="subsection-header">Advanced Data Filters</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_downs = st.multiselect(
            "Select Downs",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            help="Filter by down number"
        )
    with col2:
        play_types = st.multiselect(
            "Play Types",
            options=['run', 'pass'],
            default=['run', 'pass'],
            help="Filter by play type"
        )
    with col3:
        yard_range = st.slider(
            "Yards Gained Range", 
            int(df['yards_gained'].min()), 
            int(df['yards_gained'].max()), 
            (int(df['yards_gained'].min()), int(df['yards_gained'].max())),
            help="Filter by yards gained on play"
        )
    with col4:
        field_position = st.slider(
            "Field Position Range",
            1, 99,
            (1, 99),
            help="Distance from opponent's goal line"
        )
    
    # apply filters with progress indication
    with st.spinner("Applying filters..."):
        filtered_df = df[
            (df['down'].isin(selected_downs)) &
            (df['play_type'].isin(play_types)) &
            (df['yards_gained'] >= yard_range[0]) &
            (df['yards_gained'] <= yard_range[1]) &
            (df['distance_to_goal'] >= field_position[0]) &
            (df['distance_to_goal'] <= field_position[1])
        ]
    
    # filter results summary
    filter_percentage = len(filtered_df) / len(df) * 100
    st.markdown(f"""
    <div class="success-card">
        <strong>Filter Results:</strong> {len(filtered_df):,} plays selected ({filter_percentage:.1f}% of total dataset)
    </div>
    """, unsafe_allow_html=True)
    
    # enhanced data visualizations
    st.markdown('<div class="subsection-header">Distribution Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # yards gained distribution
        fig1 = px.histogram(
            filtered_df,
            x='yards_gained',
            nbins=40,
            title="Distribution of Yards Gained",
            color='play_type',
            barmode='overlay',
            opacity=0.7,
            color_discrete_map={'run': '#ef4444', 'pass': '#3b82f6'}
        )
        
        fig1.update_layout(
            bargap=0.1,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(font=dict(size=16)),
            legend=dict(title="Play Type")
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # box plot by down
        fig2 = px.box(
            filtered_df,
            x='down',
            y='yards_gained',
            color='play_type',
            title="Yards Distribution by Down",
            color_discrete_map={'run': '#ef4444', 'pass': '#3b82f6'}
        )
        
        fig2.update_layout(
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(font=dict(size=16)),
            legend=dict(title="Play Type")
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # correlation analysis
    st.markdown('<div class="subsection-header">Feature Correlation Analysis</div>', unsafe_allow_html=True)
    
    numeric_cols = ['down', 'ydstogo', 'distance_to_goal', 'yards_gained', 'score_diff']
    available_cols = [col for col in numeric_cols if col in filtered_df.columns]
    
    if len(available_cols) >= 2:
        corr_matrix = filtered_df[available_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )
        
        fig_corr.update_layout(
            height=400,
            font=dict(family="Inter, sans-serif"),
            title=dict(font=dict(size=16))
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # enhanced raw data viewer
    st.markdown('<div class="subsection-header">Filtered Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df[available_cols], use_container_width=True, height=400)
    csv = filtered_df[available_cols].to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="nfl_plays_filtered.csv",
        mime="text/csv",
        help="Download the displayed filtered data"
    )
    st.dataframe(filtered_df[available_cols].describe(), use_container_width=True) 