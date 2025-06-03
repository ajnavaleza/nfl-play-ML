import streamlit as st
from src.model import ExpectedYardsModel
from src.data_processing import load_and_prepare_data

@st.cache_resource
def load_model():
    try:
        model = ExpectedYardsModel()
        if model.load_model('models/expected_yards_model.pkl'):
            return model
        else:
            st.error("**Model Not Found** - No trained model detected.")
            st.info("**Getting Started:** Run `python main.py` to train the model with automatic NFL data download.")
            return None
    except Exception as e:
        st.error(f"**Model Loading Error:** {str(e)}")
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