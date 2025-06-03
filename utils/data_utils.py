import streamlit as st
from src.model import ExpectedYardsModel
from src.data_processing import load_and_prepare_data
import os

@st.cache_resource
def load_model():
    """Load model, automatically training it if it doesn't exist"""
    try:
        model = ExpectedYardsModel()
        
        # Try to load existing model first
        if os.path.exists('models/expected_yards_model.pkl'):
            if model.load_model('models/expected_yards_model.pkl'):
                return model
        
        # If model doesn't exist, train it automatically
        st.info("🏈 **First Time Setup:** Training AI model automatically...")
        trained_model = train_model_automatically()
        return trained_model
        
    except Exception as e:
        st.error(f"**Model Loading Error:** {str(e)}")
        return None

def train_model_automatically():
    """Automatically train the model with progress indicators"""
    try:
        with st.spinner("🔄 Training NFL Play Intelligence Model..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load and prepare data
            status_text.text("📊 Loading NFL data from public sources...")
            progress_bar.progress(20)
            
            X, y, feature_names, raw_df = load_and_prepare_data()
            
            # Step 2: Initialize model
            status_text.text("🤖 Initializing XGBoost model...")
            progress_bar.progress(40)
            
            model = ExpectedYardsModel(model_type='xgboost')
            
            # Step 3: Train model
            status_text.text("🎯 Training model on NFL plays...")
            progress_bar.progress(60)
            
            trained_model = model.train_model(X, y, feature_names)
            
            # Step 4: Save model
            status_text.text("💾 Saving trained model...")
            progress_bar.progress(80)
            
            os.makedirs('models', exist_ok=True)
            model.save_model('models/expected_yards_model.pkl')
            
            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("✅ Model training complete!")
            
            # Show success message
            st.success(f"""
            🎉 **Training Complete!** 
            
            **Model Stats:**
            - **Total Plays:** {len(X):,}
            - **Features:** {len(feature_names)}
            - **Pass Plays:** {(raw_df['play_type'] == 'pass').sum():,}
            - **Run Plays:** {(raw_df['play_type'] == 'run').sum():,}
            
            Your NFL Play Intelligence System is now ready!
            """)
            
            # Clear the progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return model
            
    except Exception as e:
        st.error(f"**Automatic Training Failed:** {str(e)}")
        st.markdown("""
        ### Manual Setup Option:
        If automatic training fails, you can run manually:
        ```bash
        python main.py
        ```
        Then refresh this page.
        """)
        return None

@st.cache_data
def load_sample_data():
    try:
        with st.spinner("Loading NFL analytics data..."):
            X, y, feature_names, df = load_and_prepare_data()
            
            sample_size = min(8000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            
            st.success(f"**Data Loaded:** {len(sample_df):,} plays ready for analysis")
            return sample_df
            
    except Exception as e:
        st.error(f"**Data Loading Error:** {str(e)}")
        st.info("**Data Source:** Automatically downloads from publicly available NFL repositories")
        return None

def display_setup_instructions():
    st.markdown('<div class="section-header">System Setup Required</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Quick Start Guide</h3>
            <p>To begin using the NFL Play Intelligence System, you need to train the AI model first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Step 1: Train the Model")
        st.code("python main.py", language="bash")
        
        st.markdown("### What This Does:")
        st.markdown("""
        - **Downloads** real NFL play-by-play data from public sources
        - **Processes** 20,000+ NFL plays with advanced feature engineering
        - **Trains** XGBoost machine learning model for play prediction
        - **Saves** model for instant web app access
        - **Takes** approximately 2-3 minutes to complete
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>System Requirements</h4>
            <ul>
                <li>Python 3.7+</li>
                <li>Internet connection</li>
                <li>2GB available RAM</li>
                <li>1GB disk space</li>
            </ul>
        </div>
        """, unsafe_allow_html=True) 