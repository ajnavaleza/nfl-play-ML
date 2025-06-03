"""
Scenario Simulator Page for NFL Play Intelligence System
Test AI recommendations across various game situations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_processing import get_play_features

def scenario_simulator_page(model):
    """Enhanced scenario simulator with improved UX"""
    st.markdown('<div class="section-header">Game Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("Test AI recommendations across various game situations and critical moments")
    
    # Enhanced predefined scenarios without emojis
    scenarios = {
        "Goal Line Stand": {"down": 3, "ydstogo": 2, "yardline": 3, "quarter": 4, "score_diff": -4},
        "Two Minute Drill": {"down": 2, "ydstogo": 8, "yardline": 35, "quarter": 4, "score_diff": -3},
        "Short Yardage": {"down": 4, "ydstogo": 1, "yardline": 45, "quarter": 3, "score_diff": 0},
        "Red Zone Opportunity": {"down": 1, "ydstogo": 10, "yardline": 15, "quarter": 2, "score_diff": 3},
        "Desperation Drive": {"down": 4, "ydstogo": 12, "yardline": 40, "quarter": 4, "score_diff": -7},
        "Safe Territory": {"down": 1, "ydstogo": 10, "yardline": 85, "quarter": 1, "score_diff": 7},
        "Third Down Conversion": {"down": 3, "ydstogo": 7, "yardline": 28, "quarter": 3, "score_diff": -3},
        "Opening Drive": {"down": 1, "ydstogo": 10, "yardline": 75, "quarter": 1, "score_diff": 0}
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subsection-header">Quick Scenarios</div>', unsafe_allow_html=True)
        
        selected_scenario = st.selectbox(
            "Choose a critical game situation:",
            list(scenarios.keys()),
            help="Select from common NFL game situations"
        )
        
        if st.button("Load Selected Scenario", type="primary"):
            scenario = scenarios[selected_scenario]
            for key, value in scenario.items():
                st.session_state[f"sim_{key}"] = value
            st.success(f"**Scenario Loaded:** {selected_scenario}")
        
        # Scenario description
        scenario_descriptions = {
            "Goal Line Stand": "Defense has offense pinned close to goal line. High-stakes short yardage situation.",
            "Two Minute Drill": "Team trailing, driving for potential game-tying score with limited time.",
            "Short Yardage": "Fourth down conversion attempt. Go for it or punt decision point.",
            "Red Zone Opportunity": "First down in scoring territory. Multiple play options available.",
            "Desperation Drive": "Team needs significant yardage with limited downs remaining.",
            "Safe Territory": "Conservative situation in own territory. Field position management key.",
            "Third Down Conversion": "Critical third down conversion attempt in opponent territory.",
            "Opening Drive": "Game opening drive. Setting tone and establishing rhythm."
        }
        
        if selected_scenario in scenario_descriptions:
            st.markdown(f"""
            <div class="info-card">
                <strong>Situation Analysis:</strong><br>
                {scenario_descriptions[selected_scenario]}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="subsection-header">Custom Scenario Builder</div>', unsafe_allow_html=True)
        
        # Enhanced scenario inputs with session state
        down_options = [1, 2, 3, 4]
        quarter_options = [1, 2, 3, 4]
        
        down = st.selectbox("Down", down_options, 
                           index=down_options.index(st.session_state.get('sim_down', 1)))
        
        ydstogo = st.slider("Yards to Go", 1, 25, 
                          value=st.session_state.get('sim_ydstogo', 10))
        
        yardline = st.slider("Distance to Goal", 1, 99, 
                           value=st.session_state.get('sim_yardline', 50))
        
        quarter = st.selectbox("Quarter", quarter_options,
                              index=quarter_options.index(st.session_state.get('sim_quarter', 1)))
        
        score_diff = st.slider("Score Differential", -21, 21, 
                             value=st.session_state.get('sim_score_diff', 0))
    
    with col2:
        st.markdown('<div class="subsection-header">Simulation Results</div>', unsafe_allow_html=True)
        
        # Run simulation
        features = get_play_features(down, ydstogo, yardline, quarter, score_diff)
        
        try:
            recommendation = model.recommend_play_type(features)
            
            # Enhanced results display with professional styling
            rec_play = recommendation['recommended_play'].upper()
            confidence = recommendation['confidence'].title()
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>AI Recommendation: {rec_play}</h3>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    Confidence Level: <strong>{confidence}</strong>
                </p>
                <p style="margin: 0;">
                    {recommendation['context_advice']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advantage gauge visualization
            advantage = recommendation['expected_yards_difference']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = advantage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Expected Advantage ({rec_play})", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [-3, 3], 'tickfont': {'size': 12}},
                    'bar': {'color': "#2563eb", 'thickness': 0.7},
                    'steps': [
                        {'range': [-3, -0.5], 'color': "#fecaca"},
                        {'range': [-0.5, 0.5], 'color': "#fef3c7"},
                        {'range': [0.5, 3], 'color': "#dcfce7"}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 3},
                        'thickness': 0.8,
                        'value': 0
                    }
                },
                number = {'font': {'size': 20}}
            ))
            
            fig.update_layout(
                height=300, 
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            comparison_data = [
                {
                    "Play Type": "Run Play", 
                    "Expected Yards": f"{recommendation['run_expected_yards']:.2f}",
                    "Recommended": "✓ YES" if rec_play == "RUN" else "No"
                },
                {
                    "Play Type": "Pass Play", 
                    "Expected Yards": f"{recommendation['pass_expected_yards']:.2f}",
                    "Recommended": "✓ YES" if rec_play == "PASS" else "No"
                }
            ]
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Strategic context and warnings
            st.markdown(f"""
            <div class="success-card">
                <strong>Strategic Analysis:</strong><br>
                {recommendation['context_advice']}
            </div>
            """, unsafe_allow_html=True)
            
            # Situational alerts with improved styling
            if down == 4:
                st.markdown("""
                <div class="warning-card">
                    <strong>Fourth Down Critical Decision</strong><br>
                    High-risk situation. Consider punting vs conversion attempt based on field position, score, and time remaining.
                </div>
                """, unsafe_allow_html=True)
                
            if yardline <= 5 and ydstogo >= yardline:
                st.markdown("""
                <div class="info-card">
                    <strong>Goal-to-Go Formation</strong><br>
                    Consider specialized goal line packages and short-yardage formations for maximum effectiveness.
                </div>
                """, unsafe_allow_html=True)
                
            if quarter == 4 and abs(score_diff) <= 7:
                st.markdown("""
                <div class="warning-card">
                    <strong>Crunch Time Management</strong><br>
                    Final quarter with close score. Clock management and situational awareness are critical factors.
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"**Simulation Error:** {str(e)}")
            st.info("Please verify your scenario inputs and try again.") 