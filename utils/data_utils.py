import streamlit as st
from src.model import ExpectedYardsModel
from src.data_processing import load_and_prepare_data
import os

@st.cache_resource
def load_model():
    """Load existing trained model"""
    try:
        model = ExpectedYardsModel()
        
        if os.path.exists('models/expected_yards_model.pkl'):
            if model.load_model('models/expected_yards_model.pkl'):
                return model
        
        return None
        
    except Exception as e:
        st.error(f"**Model Loading Error:** {str(e)}")
        return None

def train_model_with_progress():
    try:
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Training NFL Play Intelligence Model")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("**Step 1/5:** 📊 Loading NFL data from public sources...")
            progress_bar.progress(10)
            
            X, y, feature_names, raw_df = load_and_prepare_data()
            
            status_text.markdown("**Step 2/5:** 🤖 Initializing XGBoost model...")
            progress_bar.progress(30)
            
            model = ExpectedYardsModel(model_type='xgboost')
            
            status_text.markdown("**Step 3/5:** 🎯 Training model on NFL plays...")
            progress_bar.progress(50)
            
            trained_model = model.train_model(X, y, feature_names)
            
            status_text.markdown("**Step 4/5:** 💾 Saving trained model...")
            progress_bar.progress(80)
            
            os.makedirs('models', exist_ok=True)
            model.save_model('models/expected_yards_model.pkl')
            
            status_text.markdown("**Step 5/5:** ✅ Finalizing setup...")
            progress_bar.progress(100)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"""
            **Training Complete!** 
            
            **Model Performance:**
            - **Total Plays Analyzed:** {len(X):,}
            - **Features Engineered:** {len(feature_names)}
            - **Pass Plays:** {(raw_df['play_type'] == 'pass').sum():,} ({(raw_df['play_type'] == 'pass').mean():.1%})
            - **Run Plays:** {(raw_df['play_type'] == 'run').sum():,} ({(raw_df['play_type'] == 'run').mean():.1%})
            
            **🚀 Your NFL Play Intelligence System is now ready!**
            
            Click any page in the sidebar to start analyzing plays.
            """)
            
            st.cache_resource.clear()
            
            return True
            
    except Exception as e:
        st.error(f"**Training Failed:** {str(e)}")
        st.markdown("""
        ### Troubleshooting:
        - Check your internet connection (required for NFL data download)
        - Ensure all dependencies are installed: `pip install -r requirements.txt`
        - Try refreshing the page and clicking "Train Model" again
        """)
        return False

@st.cache_data
def load_data():
    try:
        with st.spinner("Loading NFL analytics data..."):
            X, y, feature_names, df = load_and_prepare_data()
            if df is None or len(df) == 0:
                st.error("**Data Loading Error:** No valid data could be loaded")
                st.info("Please check your internet connection and try again")
                return None
            st.success(f"**Data Loaded:** {len(df):,} plays ready for analysis (full dataset)")
            return df
    except Exception as e:
        st.error(f"**Data Loading Error:** {str(e)}")
        st.info("""
        **Data Source:** NFL play-by-play data from nflfastR
        If you're seeing this error:
        1. Check your internet connection
        2. Try refreshing the page
        3. If the problem persists, the data source might be temporarily unavailable
        """)
        return None

def display_setup_instructions():
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(45deg, #2563eb, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏈 NFL Play Intelligence System
        </h1>
        <p style="font-size: 1.25rem; color: #64748b; margin-bottom: 2rem;">
            Machine Learning-Powered Play Calling Intelligence • Powered by Real NFL Data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2>What This App Does</h2>
            <p>The NFL Play Intelligence System uses machine learning to analyze real NFL play-by-play data and provide intelligent recommendations for play calling based on game situations.</p>
            <ul>
                <li>Play Predictor: Get AI recommendations for run vs pass based on down, distance, field position, and game context</li>
                <li>Analytics Dashboard: Explore comprehensive NFL statistics and trends</li>
                <li>Model Insights: Understand which factors most influence play success</li>
                <li>Scenario Simulator: Test different game situations and see expected outcomes</li>
                <li>Data Explorer: Dive deep into real NFL play-by-play data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>How It Works</h3>
            <ol>
                <li>Data Collection: Downloads real NFL play-by-play data from nflfastR (public repository)</li>
                <li>Feature Engineering: Creates 37 advanced features including situational context, field position, and game state</li>
                <li>Model Training: Uses XGBoost regression to predict expected yards for run vs pass plays</li>
                <li>Intelligent Recommendations: Provides contextual advice based on predicted outcomes</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="success-card">
            <h3>Model Specifications</h3>
            <ul>
                <li><strong>Data Source:</strong> nflfastR Public API</li>
                <li><strong>Seasons:</strong> 2019-2021 NFL Data</li>
                <li><strong>Data Volume:</strong> 20,000+ Real Plays</li>
                <li><strong>Algorithm:</strong> XGBoost Regression</li>
                <li><strong>Features:</strong> 37 Engineered Variables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-card">
            <h3>⚡ Quick Start</h3>
            <p><strong>First Time Setup:</strong></p>
            <p>Click the "Train Model" button below to:</p>
            <ul>
                <li>📥 Download real NFL data</li>
                <li>🧠 Train the AI model</li>
                <li>💾 Save for instant access</li>
            </ul>
            <p><strong>Time Required:</strong> 2-3 minutes</p>
            <p><strong>Internet Required:</strong> Yes (for data download)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training Section
    st.markdown("---")
    st.markdown("### 🚀 Ready to Get Started?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Train Model", type="primary", help="Start the AI model training process", use_container_width=True):
            st.markdown("---")
            success = train_model_with_progress()
            if success:
                st.balloons()
                st.rerun()
