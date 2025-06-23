import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_utils import load_data

def analytics_dashboard_page():
    st.markdown('<div class="section-header">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of NFL play-calling trends and effectiveness")
    
    # key performance indicators
    st.markdown('<div class="subsection-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_yards = df['yards_gained'].mean()
        st.metric(
            "Average Yards per Play",
            f"{avg_yards:.2f}",
            delta=f"{avg_yards - 4.8:.2f} vs NFL average"
        )
    
    with col2:
        pass_pct = (df['play_type'] == 'pass').mean() * 100
        st.metric(
            "Pass Play Percentage",
            f"{pass_pct:.1f}%",
            delta=f"{pass_pct - 58:.1f}% vs typical"
        )
    
    with col3:
        third_down_plays = df[df['down'] == 3]
        if len(third_down_plays) > 0:
            third_down_success = ((third_down_plays['yards_gained'] >= third_down_plays['ydstogo']).sum() / 
                                len(third_down_plays))
        else:
            third_down_success = 0
        st.metric("Third Down Success Rate", f"{third_down_success:.1%}")
    
    with col4:
        red_zone_plays = df[df['red_zone'] == 1] if 'red_zone' in df.columns else df[df['distance_to_goal'] <= 20]
        red_zone_avg = red_zone_plays['yards_gained'].mean() if len(red_zone_plays) > 0 else 0
        st.metric("Red Zone Average", f"{red_zone_avg:.2f} yards")
    
    # performance analysis charts
    st.markdown('<div class="subsection-header">Performance Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # yards by down analysis
        down_stats = df.groupby('down').agg({
            'yards_gained': ['mean', 'count']
        }).round(2)
        down_stats.columns = ['Average Yards', 'Play Count']
        
        fig1 = px.bar(
            down_stats.reset_index(),
            x='down',
            y='Average Yards',
            title="Average Yards Gained by Down",
            color='Average Yards',
            color_continuous_scale='Blues',
            text='Average Yards'
        )
        
        fig1.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside',
            textfont=dict(size=12)
        )
        
        fig1.update_layout(
            showlegend=False,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(font=dict(size=16))
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # play type effectiveness
        play_effectiveness = df.groupby(['down', 'play_type'])['yards_gained'].mean().reset_index()
        
        fig2 = px.bar(
            play_effectiveness,
            x='down',
            y='yards_gained',
            color='play_type',
            barmode='group',
            title="Effectiveness by Down and Play Type",
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
    
    # field position impact analysis
    st.markdown('<div class="subsection-header">Field Position Impact</div>', unsafe_allow_html=True)
    
    # create field zones for analysis
    df['field_zone'] = pd.cut(
        df['distance_to_goal'],
        bins=[0, 10, 20, 40, 60, 100],
        labels=['Goal Line (1-10)', 'Red Zone (11-20)', 'Short Field (21-40)', 'Mid Field (41-60)', 'Long Field (61+)']
    )
    
    zone_analysis = df.groupby(['field_zone', 'play_type']).agg({
        'yards_gained': ['mean', 'count']
    }).round(2)
    
    zone_analysis.columns = ['Average Yards', 'Play Count']
    zone_analysis = zone_analysis.reset_index()
    
    fig3 = px.bar(
        zone_analysis,
        x='field_zone',
        y='Average Yards',
        color='play_type',
        barmode='group',
        title="Expected Yards by Field Position Zone",
        color_discrete_map={'run': '#ef4444', 'pass': '#3b82f6'}
    )
    
    fig3.update_layout(
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(font=dict(size=16)),
        legend=dict(title="Play Type")
    )
    
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)
    
    # success rate analysis
    st.markdown('<div class="subsection-header">Success Rate Analysis</div>', unsafe_allow_html=True)
    
    # define success as gaining required yards
    df['success'] = df['yards_gained'] >= df['ydstogo']
    
    success_by_situation = df.groupby(['down', 'play_type'])['success'].agg(['mean', 'count']).reset_index()
    success_by_situation.columns = ['down', 'play_type', 'success_rate', 'total_plays']
    
    fig4 = px.bar(
        success_by_situation,
        x='down',
        y='success_rate',
        color='play_type',
        barmode='group',
        title="Play Success Rate by Down",
        color_discrete_map={'run': '#ef4444', 'pass': '#3b82f6'}
    )
    
    fig4.update_layout(
        yaxis_title="Success Rate",
        yaxis_tickformat='.1%',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(font=dict(size=16)),
        legend=dict(title="Play Type")
    )
    
    st.plotly_chart(fig4, use_container_width=True) 