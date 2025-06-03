"""
Streamlit CSS Styles for NFL Play Intelligence System
Professional styling and accessibility improvements
"""

import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #2563eb;
            --primary-light: #3b82f6;
            --primary-dark: #1d4ed8;
            --secondary-color: #059669;
            --secondary-light: #10b981;
            --accent-color: #dc2626;
            --accent-light: #ef4444;
            --neutral-50: #f8fafc;
            --neutral-100: #f1f5f9;
            --neutral-200: #e2e8f0;
            --neutral-300: #cbd5e1;
            --neutral-600: #475569;
            --neutral-700: #334155;
            --neutral-800: #1e293b;
            --neutral-900: #0f172a;
            --success-color: #059669;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --info-color: #0284c7;
        }
        
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--neutral-50);
        }
        
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--neutral-900);
            text-align: center;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        
        .main-subtitle {
            font-size: 1.125rem;
            color: var(--neutral-600);
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Sidebar improvements */
        .css-1d391kg {
            background-color: var(--neutral-100);
            border-right: 2px solid var(--neutral-200);
        }
        
        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--neutral-800);
            margin-bottom: 1rem;
        }
        
        /* Card components */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, var(--neutral-50) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--neutral-200);
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.15);
        }
        
        .recommendation-card {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px 0 rgba(37, 99, 235, 0.25);
            margin: 1.5rem 0;
        }
        
        .success-card {
            background: linear-gradient(135deg, var(--success-color) 0%, var(--secondary-light) 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px 0 rgba(5, 150, 105, 0.2);
            margin: 1rem 0;
        }
        
        .warning-card {
            background: linear-gradient(135deg, var(--warning-color) 0%, #f59e0b 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px 0 rgba(217, 119, 6, 0.2);
            margin: 1rem 0;
        }
        
        .info-card {
            background: linear-gradient(135deg, var(--info-color) 0%, #0ea5e9 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px 0 rgba(2, 132, 199, 0.2);
            margin: 1rem 0;
        }
        
        /* Button improvements */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px 0 rgba(37, 99, 235, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px 0 rgba(37, 99, 235, 0.3);
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--neutral-800);
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--neutral-200);
        }
        
        .subsection-header {
            font-size: 1.25rem;
            font-weight: 500;
            color: var(--neutral-700);
            margin: 1.5rem 0 0.75rem 0;
        }
        
        /* Metric styling */
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--neutral-200);
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        /* Data display improvements */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        /* Alert components */
        .stAlert {
            border-radius: 8px;
            border: none;
            font-weight: 500;
        }
        
        /* Chart improvements */
        .js-plotly-plot {
            border-radius: 8px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        /* Navigation improvements */
        .nav-link {
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            color: var(--neutral-700);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .nav-link:hover {
            background-color: var(--primary-color);
            color: white;
        }
        
        /* Accessibility improvements */
        .stSelectbox label, .stSlider label, .stMultiSelect label {
            font-weight: 500;
            color: var(--neutral-700);
            margin-bottom: 0.5rem;
        }
        
        /* Focus indicators */
        .stButton > button:focus,
        .stSelectbox > div > div:focus,
        .stSlider > div > div > div > div:focus {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            :root {
                --primary-color: #1e40af;
                --secondary-color: #047857;
                --neutral-600: #374151;
                --neutral-700: #1f2937;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            .metric-card, .stButton > button {
                transition: none;
            }
            
            .metric-card:hover, .stButton > button:hover {
                transform: none;
            }
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-success { background-color: var(--success-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-error { background-color: var(--error-color); }
        .status-info { background-color: var(--info-color); }
        
        /* Responsive improvements */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .metric-card, .recommendation-card {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True) 