import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from src.model import ExpectedYardsModel
from src.data_processing import get_play_context_features, load_data, clean_data, feature_engineering
import os

# Page configuration
st.set_page_config(
    page_title="🏈 Expected Yards & Play Calling Intelligence System",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = ExpectedYardsModel()
        if model.load_model('models/expected_yards_model.pkl'):
            return model
        else:
            st.error("Failed to load model. Please train the model first by running 'python main.py'")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for analytics"""
    try:
        df = load_data('data/nfl_pbp_2021_2022.csv')
        df = clean_data(df)
        df = feature_engineering(df)
        return df.sample(n=10000, random_state=42) 
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">🏈 Expected Yards & Play Calling Intelligence System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🏈 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎯 Play Predictor", "📊 Analytics Dashboard", "🧠 Model Insights", "🎮 Scenario Simulator", "📈 Data Explorer"]
    )
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Route to different pages
    if page == "🎯 Play Predictor":
        play_predictor_page(model)
    elif page == "📊 Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "🧠 Model Insights":
        model_insights_page(model)
    elif page == "🎮 Scenario Simulator":
        scenario_simulator_page(model)
    elif page == "📈 Data Explorer":
        data_explorer_page()

def play_predictor_page(model):
    # play predictor page
    st.header("🎯 Play Predictor & Recommender")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Game Situation")
        
        # Game context inputs
        down = st.selectbox("Down", [1, 2, 3, 4], index=1)
        ydstogo = st.slider("Yards to Go", 1, 25, 10)
        yardline = st.slider("Yards from Goal", 1, 99, 50, 
                           help="Distance to opponent's end zone")
        
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=2)
        score_diff = st.slider("Score Differential", -21, 21, 0, 
                             help="Positive if leading, negative if trailing")
    
    with col2:
        st.subheader("Prediction Results")
        
        # create features dictionary
        features = get_play_context_features(down, ydstogo, yardline, quarter, score_diff)
        
        # get recommendation
        try:
            recommendation = model.recommend_play_type(features)
            
            # display recommendation
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown(f"### Recommended Play: **{recommendation['recommended_play'].upper()}**")
            
            col_run, col_pass = st.columns(2)
            with col_run:
                st.metric("Run Expected Yards", f"{recommendation['run_expected_yards']:.2f}")
            with col_pass:
                st.metric("Pass Expected Yards", f"{recommendation['pass_expected_yards']:.2f}")
            
            st.markdown(f"**Confidence:** {recommendation['confidence'].title()}")
            st.markdown(f"**Context Advice:** {recommendation['context_advice']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # visualization
            fig = go.Figure(data=[
                go.Bar(name='Expected Yards', 
                      x=['Run', 'Pass'], 
                      y=[recommendation['run_expected_yards'], recommendation['pass_expected_yards']],
                      marker_color=['#ff6b6b', '#4ecdc4'])
            ])
            fig.update_layout(
                title="Expected Yards Comparison",
                yaxis_title="Expected Yards",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def analytics_dashboard_page():
    # analytics dashboard page
    st.header("📊 Analytics Dashboard")
    
    # load sample data
    df = load_sample_data()
    if df is None:
        return
    
    # key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_yards = df['yards_gained'].mean()
        st.metric("Average Yards/Play", f"{avg_yards:.2f}")
    
    with col2:
        pass_pct = (df['play_type'] == 'pass').mean() * 100
        st.metric("Pass Play %", f"{pass_pct:.1f}%")
    
    with col3:
        third_down_conv = df[df['down'] == 3]['yards_gained'].apply(
            lambda x: x >= df[df['down'] == 3]['ydstogo'].iloc[0] 
            if len(df[df['down'] == 3]) > 0 else False
        ).mean() if len(df[df['down'] == 3]) > 0 else 0
        st.metric("3rd Down Conv. Rate", f"{third_down_conv:.1%}")
    
    with col4:
        red_zone_td = df[df['red_zone'] == 1]['yards_gained'].apply(
            lambda x: x >= 20
        ).mean() if len(df[df['red_zone'] == 1]) > 0 else 0
        st.metric("Red Zone TD Rate", f"{red_zone_td:.1%}")
    
    # charts
    col1, col2 = st.columns(2)
    
    with col1:
        # yards by down
        down_yards = df.groupby('down')['yards_gained'].mean().reset_index()
        fig1 = px.bar(down_yards, x='down', y='yards_gained', 
                     title="Average Yards by Down",
                     color='yards_gained', color_continuous_scale='viridis')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # yards by play type and down
        play_down_yards = df.groupby(['down', 'play_type'])['yards_gained'].mean().reset_index()
        fig2 = px.bar(play_down_yards, x='down', y='yards_gained', 
                     color='play_type', barmode='group',
                     title="Average Yards by Down and Play Type")
        st.plotly_chart(fig2, use_container_width=True)
    
    # field position analysis
    st.subheader("Field Position Analysis")
    
    # create field position bins
    df['field_position_bin'] = pd.cut(df['distance_to_goal'], 
                                     bins=[0, 10, 20, 40, 60, 100], 
                                     labels=['Goal Line (0-10)', 'Red Zone (11-20)', 
                                            'Short Field (21-40)', 'Mid Field (41-60)', 
                                            'Long Field (61-100)'])
    
    field_analysis = df.groupby(['field_position_bin', 'play_type'])['yards_gained'].mean().reset_index()
    fig3 = px.bar(field_analysis, x='field_position_bin', y='yards_gained', 
                 color='play_type', barmode='group',
                 title="Expected Yards by Field Position and Play Type")
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

def model_insights_page(model):
    # model explainability and insights
    st.header("🧠 Model Insights & Explainability")
    
    # feature importance
    st.subheader("Feature Importance")
    importance = model.get_feature_importance()
    
    if importance:
        # create dataframe for plotting
        feature_df = pd.DataFrame(list(importance.items())[:15], 
                                columns=['Feature', 'Importance'])
        
        fig = px.bar(feature_df, x='Importance', y='Feature', 
                    orientation='h', title="Top 15 Most Important Features")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # feature descriptions
        with st.expander("Feature Descriptions"):
            feature_descriptions = {
                'down': 'Current down (1-4)',
                'ydstogo': 'Yards needed for first down',
                'distance_to_goal': 'Yards from opponent goal line',
                'is_pass': 'Whether the play is a pass (1) or run (0)',
                'red_zone': 'Whether the play is in the red zone (≤20 yards)',
                'third_down': 'Whether it is 3rd down',
                'fourth_down': 'Whether it is 4th down',
                'short_yardage': 'Whether it is short yardage (≤3 yards to go)',
                'long_yardage': 'Whether it is long yardage (≥8 yards to go)',
                'score_diff': 'Score differential (positive if leading)',
                'fourth_quarter': 'Whether it is the 4th quarter',
                'first_half': 'Whether it is the first half'
            }
            
            for feature, description in feature_descriptions.items():
                if feature in importance:
                    st.write(f"**{feature}**: {description}")
    
    # SHAP explanation for sample prediction
    st.subheader("SHAP Explanation Example")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Configure a play scenario to see SHAP explanations:")
        sample_down = st.selectbox("Sample Down", [1, 2, 3, 4], key="shap_down")
        sample_ydstogo = st.slider("Sample Yards to Go", 1, 25, 10, key="shap_ydstogo")
        sample_yardline = st.slider("Sample Yards from Goal", 1, 99, 50, key="shap_yardline")
        play_type = st.selectbox("Play Type", ["pass", "run"], key="shap_play_type")
    
    with col2:
        # get SHAP explanation
        features = get_play_context_features(sample_down, sample_ydstogo, sample_yardline)
        
        try:
            explanation = model.explain_prediction(features, play_type)
            predicted_yards = model.predict_expected_yards(features, play_type)
            
            st.write(f"**Predicted Expected Yards**: {predicted_yards:.2f}")
            
            # display top SHAP values
            st.write("**Top Contributing Factors:**")
            
            shap_df = pd.DataFrame([
                {
                    'Feature': feature,
                    'Value': data['value'],
                    'SHAP Value': data['shap_value'],
                    'Impact': 'Increases' if data['shap_value'] > 0 else 'Decreases'
                }
                for feature, data in list(explanation.items())[:8]
            ])
            
            # color code the SHAP values
            def color_shap_values(val):
                if val > 0:
                    return 'background-color: #d4edda'
                else:
                    return 'background-color: #f8d7da'
            
            styled_df = shap_df.style.applymap(color_shap_values, subset=['SHAP Value'])
            st.dataframe(styled_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {e}")

def scenario_simulator_page(model):
    # game scenario simulator
    st.header("🎮 Scenario Simulator")
    
    st.write("Simulate different game scenarios and see how play recommendations change.")
    
    # predefined scenarios
    scenarios = {
        "Goal Line Stand": {"down": 3, "ydstogo": 2, "yardline": 3, "quarter": 4, "score_diff": -4},
        "Two Minute Drill": {"down": 2, "ydstogo": 8, "yardline": 35, "quarter": 4, "score_diff": -3},
        "Short Yardage": {"down": 4, "ydstogo": 1, "yardline": 45, "quarter": 3, "score_diff": 0},
        "Red Zone": {"down": 1, "ydstogo": 10, "yardline": 15, "quarter": 2, "score_diff": 3},
        "Hail Mary": {"down": 4, "ydstogo": 12, "yardline": 40, "quarter": 4, "score_diff": -7}
    }
    
    selected_scenario = st.selectbox("Choose a scenario:", list(scenarios.keys()))
    
    if st.button("Load Scenario"):
        scenario = scenarios[selected_scenario]
        st.session_state.update(scenario)
    
    # Scenario inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # calculate indices for default values
        down_options = [1, 2, 3, 4]
        quarter_options = [1, 2, 3, 4]
        
        down_default = st.session_state.get('down', 1)
        quarter_default = st.session_state.get('quarter', 1)
        
        down_index = down_options.index(down_default) if down_default in down_options else 0
        quarter_index = quarter_options.index(quarter_default) if quarter_default in quarter_options else 0
        
        down = st.selectbox("Down", down_options, index=down_index)
        ydstogo = st.slider("Yards to Go", 1, 25, 
                          value=st.session_state.get('ydstogo', 10))
        yardline = st.slider("Yards from Goal", 1, 99, 
                           value=st.session_state.get('yardline', 50))
        quarter = st.selectbox("Quarter", quarter_options, index=quarter_index)
        score_diff = st.slider("Score Differential", -21, 21, 
                             value=st.session_state.get('score_diff', 0))
    
    with col2:
        # run simulation
        features = get_play_context_features(down, ydstogo, yardline, quarter, score_diff)
        
        try:
            recommendation = model.recommend_play_type(features)
            
            st.markdown("### Simulation Results")
            
            # create gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = recommendation['expected_yards_difference'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Yards Difference (Pass - Run)"},
                gauge = {
                    'axis': {'range': [-5, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-5, -1], 'color': "lightgray"},
                        {'range': [-1, 1], 'color': "gray"},
                        {'range': [1, 5], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # results table
            results_df = pd.DataFrame([
                {"Play Type": "Run", "Expected Yards": f"{recommendation['run_expected_yards']:.2f}"},
                {"Play Type": "Pass", "Expected Yards": f"{recommendation['pass_expected_yards']:.2f}"}
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            st.success(f"**Recommendation**: {recommendation['recommended_play'].upper()}")
            st.info(f"**Strategic Advice**: {recommendation['context_advice']}")
            
        except Exception as e:
            st.error(f"Error running simulation: {e}")

def data_explorer_page():
    # data exploration interface
    st.header("📈 Data Explorer")
    
    # load sample data
    df = load_sample_data()
    if df is None:
        return
    
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Plays", f"{len(df):,}")
    with col2:
        st.metric("Unique Teams", df['posteam'].nunique() if 'posteam' in df.columns else "N/A")
    with col3:
        st.metric("Time Period", "2021-2022")
    
    # data filters
    st.subheader("Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_downs = st.multiselect("Select Downs", [1, 2, 3, 4], default=[1, 2, 3, 4])
    with col2:
        play_types = st.multiselect("Select Play Types", ['run', 'pass'], default=['run', 'pass'])
    with col3:
        yard_range = st.slider("Yards Gained Range", 
                              int(df['yards_gained'].min()), 
                              int(df['yards_gained'].max()), 
                              (int(df['yards_gained'].min()), int(df['yards_gained'].max())))
    
    # apply filters
    filtered_df = df[
        (df['down'].isin(selected_downs)) &
        (df['play_type'].isin(play_types)) &
        (df['yards_gained'] >= yard_range[0]) &
        (df['yards_gained'] <= yard_range[1])
    ]
    
    st.write(f"Filtered dataset: {len(filtered_df):,} plays")
    
    # distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(filtered_df, x='yards_gained', nbins=30, 
                           title="Distribution of Yards Gained",
                           color='play_type')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.box(filtered_df, x='down', y='yards_gained', 
                     color='play_type',
                     title="Yards Gained by Down and Play Type")
        st.plotly_chart(fig2, use_container_width=True)
    
    # raw data viewer
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data Sample")
        display_cols = ['down', 'ydstogo', 'yardline_100', 'play_type', 'yards_gained']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        st.dataframe(filtered_df[available_cols].head(100), use_container_width=True)

if __name__ == "__main__":
    main() 