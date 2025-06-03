import streamlit as st
from utils.styles import apply_custom_styles
from utils.data_utils import load_model, display_setup_instructions
from page_modules.play_predictor import play_predictor_page
from page_modules.analytics_dashboard import analytics_dashboard_page
from page_modules.model_insights import model_insights_page
from page_modules.scenario_simulator import scenario_simulator_page
from page_modules.data_explorer import data_explorer_page

st.set_page_config(
    page_title="NFL Play Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "NFL Expected Yards & Play Calling Intelligence System - AI-powered play recommendations for optimal offensive strategy."
    }
)

apply_custom_styles()

def render_header():
    st.markdown("""
    <div class="main-header">NFL Play Intelligence System</div>
    <div class="main-subtitle">
        Machine Learning-Powered Play Calling Intelligence • Powered by Real NFL Data
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
        
        page = st.selectbox(
            "Select Analysis Page:",
            [
                "Play Predictor",
                "Analytics Dashboard", 
                "Model Insights",
                "Scenario Simulator",
                "Data Explorer"
            ],
            help="Choose the analysis section you want to explore"
        )
        
        st.markdown("---")
        
        st.markdown("### Data Information")
        st.markdown("""
        **Source:** nflfastR Public Repository  
        **Coverage:** NFL Seasons 2019-2021  
        **Size:** Automatically downloaded  
        **Updates:** Real-time from public APIs
        """)
        
        st.markdown("---")
        
        st.markdown("### System Status")
        model = load_model()
        if model:
            st.markdown('<span class="status-indicator status-success"></span>**Model:** Ready', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-error"></span>**Model:** Training Required', unsafe_allow_html=True)
        
        return page, model

def route_to_page(page, model):
    if model is None and page != "Data Explorer":
        display_setup_instructions()
        return
    
    try:
        if page == "Play Predictor":
            play_predictor_page(model)
        elif page == "Analytics Dashboard":
            analytics_dashboard_page()
        elif page == "Model Insights":
            model_insights_page(model)
        elif page == "Scenario Simulator":
            scenario_simulator_page(model)
        elif page == "Data Explorer":
            data_explorer_page()
    except Exception as e:
        st.error(f"**Page Error:** {str(e)}")
        st.info("Please refresh the page or try a different section.")

def main():
    render_header()
    page, model = render_sidebar()
    
    route_to_page(page, model)

if __name__ == "__main__":
    main() 