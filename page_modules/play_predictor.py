"""
Play Predictor Page for NFL Play Intelligence System
AI-powered play recommendations based on game situation
"""

import streamlit as st
import plotly.graph_objects as go
from src.data_processing import get_play_features

def play_predictor_page(model):
    """Enhanced play predictor with improved UX"""
    st.markdown('<div class="section-header">Play Predictor & Recommender</div>', unsafe_allow_html=True)
    st.markdown("Get AI-powered play recommendations based on current game situation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subsection-header">Game Situation</div>', unsafe_allow_html=True)
        
        # Enhanced input controls with better descriptions
        down = st.selectbox(
            "Current Down",
            options=[1, 2, 3, 4],
            index=1,
            help="Which down is the offense currently on (1st through 4th down)"
        )
        
        ydstogo = st.slider(
            "Yards to Go",
            min_value=1,
            max_value=25,
            value=10,
            help="Yards needed to achieve a first down or touchdown"
        )
        
        yardline = st.slider(
            "Distance to Goal Line",
            min_value=1,
            max_value=99,
            value=50,
            help="Yards remaining to the opponent's end zone"
        )
        
        quarter = st.selectbox(
            "Current Quarter",
            options=[1, 2, 3, 4],
            index=1,
            help="Which quarter of the game is currently being played"
        )
        
        score_diff = st.slider(
            "Score Differential",
            min_value=-21,
            max_value=21,
            value=0,
            help="Your team's score minus opponent's score (positive = leading)"
        )
        
        # Contextual information cards
        if yardline <= 20:
            st.markdown("""
            <div class="success-card">
                <strong>Red Zone Opportunity</strong><br>
                High-percentage scoring situation. Consider goal-line plays.
            </div>
            """, unsafe_allow_html=True)
        elif yardline >= 80:
            st.markdown("""
            <div class="info-card">
                <strong>Own Territory</strong><br>
                Field position matters. Conservative play calling recommended.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">AI Recommendation</div>', unsafe_allow_html=True)
        
        # Generate recommendation
        features = get_play_features(down, ydstogo, yardline, quarter, score_diff)
        
        try:
            recommendation = model.recommend_play_type(features)
            
            # Main recommendation display
            rec_play = recommendation['recommended_play'].upper()
            confidence = recommendation['confidence'].title()
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>Recommended Play: {rec_play}</h3>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    Confidence Level: <strong>{confidence}</strong>
                </p>
                <p style="margin: 0;">
                    {recommendation['context_advice']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Expected yards comparison
            col_metrics1, col_metrics2 = st.columns(2)
            
            with col_metrics1:
                st.metric(
                    "Run Expected Yards",
                    f"{recommendation['run_expected_yards']:.2f}",
                    delta=f"{recommendation['run_expected_yards'] - 4.5:.2f}" if rec_play == "RUN" else None
                )
            
            with col_metrics2:
                st.metric(
                    "Pass Expected Yards",
                    f"{recommendation['pass_expected_yards']:.2f}",
                    delta=f"{recommendation['pass_expected_yards'] - 6.8:.2f}" if rec_play == "PASS" else None
                )
            
            # Expected yards advantage
            advantage = abs(recommendation['expected_yards_difference'])
            st.metric(
                "Predicted Advantage",
                f"{advantage:.2f} yards",
                delta=f"+{advantage:.2f}" if advantage > 0 else None
            )
            
            # Enhanced visualization
            fig = go.Figure()
            
            colors = ['#ef4444' if rec_play == 'RUN' else '#fca5a5', 
                     '#3b82f6' if rec_play == 'PASS' else '#93c5fd']
            
            fig.add_trace(go.Bar(
                name='Expected Yards',
                x=['Run Play', 'Pass Play'],
                y=[recommendation['run_expected_yards'], recommendation['pass_expected_yards']],
                marker_color=colors,
                text=[f"{recommendation['run_expected_yards']:.2f}", 
                      f"{recommendation['pass_expected_yards']:.2f}"],
                textposition='auto',
                textfont=dict(size=14, color='white'),
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Expected Yards Comparison - {rec_play} Recommended",
                    font=dict(size=16, color='#1e293b')
                ),
                yaxis_title="Expected Yards Gained",
                showlegend=False,
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif"),
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            fig.update_xaxes(tickfont=dict(size=12))
            fig.update_yaxes(tickfont=dict(size=12))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Situational alerts
            if down == 4:
                st.markdown("""
                <div class="warning-card">
                    <strong>Fourth Down Decision</strong><br>
                    High-risk situation. Consider field position, score, and time remaining.
                </div>
                """, unsafe_allow_html=True)
            
            if yardline <= 5 and ydstogo >= yardline:
                st.markdown("""
                <div class="info-card">
                    <strong>Goal-to-Go Situation</strong><br>
                    Consider specialized red zone formations and short-yardage plays.
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"**Prediction Error:** {str(e)}")
            st.info("Please check your inputs and try again.") 