import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_processing import get_play_features

def model_insights_page(model):
    st.markdown('<div class="section-header">Model Insights & Explainability</div>', unsafe_allow_html=True)
    st.markdown("Understand how the AI makes play recommendations with feature analysis")
    
    # feature importance analysis
    st.markdown('<div class="subsection-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    importance = model.get_feature_importance()
    
    if importance:
        # create enhanced feature importance visualization
        feature_df = pd.DataFrame(list(importance.items())[:20], 
                                columns=['Feature', 'Importance'])
        
        # add feature categories with improved categorization
        def categorize_feature(feature):
            if feature in ['down', 'ydstogo', 'distance_to_goal']:
                return 'Core Situation'
            elif 'down' in feature or 'yardage' in feature:
                return 'Down & Distance'
            elif 'zone' in feature or 'goal' in feature or 'territory' in feature:
                return 'Field Position'
            elif 'score' in feature or 'losing' in feature or 'winning' in feature:
                return 'Game Context'
            elif 'quarter' in feature or 'half' in feature:
                return 'Time Context'
            else:
                return 'Advanced Features'
        
        feature_df['Category'] = feature_df['Feature'].apply(categorize_feature)
        
        # enhanced visualization with professional color scheme
        fig = px.bar(
            feature_df, 
            x='Importance', 
            y='Feature', 
            color='Category', 
            orientation='h',
            title="Top 20 Most Important Features for Play Prediction",
            color_discrete_sequence=['#2563eb', '#059669', '#dc2626', '#d97706', '#7c3aed', '#0284c7']
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
            height=800,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title=dict(font=dict(size=18, color='#1e293b')),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # enhanced feature descriptions with professional styling
        with st.expander("Feature Dictionary & Explanations"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Core Situation Features")
                st.markdown("""
                - **down**: Current down (1st, 2nd, 3rd, or 4th)
                - **ydstogo**: Yards needed for first down or touchdown
                - **distance_to_goal**: Yards remaining to opponent's goal line
                """)
                
                st.markdown("### Field Position Features")
                st.markdown("""
                - **red_zone**: Within 20 yards of goal (high scoring probability)
                - **goal_line**: Within 5 yards of goal (touchdown likely)
                - **own_territory**: Beyond 50-yard line (conservative play calling)
                """)
            
            with col2:
                st.markdown("### Game Context Features")
                st.markdown("""
                - **score_diff**: Point differential (positive = team leading)
                - **close_game**: Score within 7 points (affects play selection)
                - **blowout**: Score difference 14+ points (affects strategy)
                """)
                
                st.markdown("### Time Context Features")
                st.markdown("""
                - **quarter**: Current quarter (1-4, affects urgency)
                - **two_minute_warning**: Final 2 minutes (clock management critical)
                - **fourth_quarter**: Final quarter (affects risk tolerance)
                """)
    
    # interactive prediction analysis
    st.markdown('<div class="subsection-header">Interactive Play Analysis</div>', unsafe_allow_html=True)
    st.markdown("Analyze how different game situations influence AI recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Configure Analysis Scenario:**")
        
        input_down = st.selectbox("Down", [1, 2, 3, 4], key="insights_down")
        input_ydstogo = st.slider("Yards to Go", 1, 25, 10, key="insights_ydstogo")
        input_yardline = st.slider("Distance to Goal", 1, 99, 50, key="insights_yardline")
        input_quarter = st.selectbox("Quarter", [1, 2, 3, 4], key="insights_quarter")
        input_score = st.slider("Score Differential", -21, 21, 0, key="insights_score")
        play_type = st.selectbox("Analyze Play Type", ["pass", "run"], key="insights_play_type")
    
    with col2:
        # generate explanation
        features = get_play_features(input_down, input_ydstogo, input_yardline, input_quarter, input_score)
        
        try:
            explanation = model.explain_prediction(features, play_type)
            predicted_yards = model.predict_expected_yards(features, play_type)
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>Prediction Analysis</h3>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                    Expected Yards: <strong>{predicted_yards:.2f}</strong>
                </p>
                <p>Play Type: <strong>{play_type.upper()}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # feature impact analysis
            explanation_data = []
            for feature, data in list(explanation.items())[:10]:
                explanation_data.append({
                    'Feature': feature.replace('_', ' ').title(),
                    'Value': data['value'],
                    'Impact Score': f"{data['importance_score']:.3f}",
                    'Effect': 'Positive Impact' if data['importance_score'] > 0 else 'Negative Impact'
                })
            
            explanation_df = pd.DataFrame(explanation_data)
            
            # professional styling for the dataframe
            def style_impact(val):
                try:
                    impact = float(val)
                    if impact > 0:
                        return 'background-color: #dcfce7; color: #166534; font-weight: 500;'
                    else:
                        return 'background-color: #fecaca; color: #991b1b; font-weight: 500;'
                except:
                    return ''
            
            styled_df = explanation_df.style.applymap(style_impact, subset=['Impact Score'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
        except Exception as e:
            st.error(f"**Analysis Error:** {str(e)}")
            st.info("This feature uses advanced model interpretability techniques. Ensure model is properly trained.") 