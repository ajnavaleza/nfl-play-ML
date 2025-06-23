# NFL Play Intelligence System - Project Structure

## Overview
The NFL Play Intelligence System has been restructured into a modular architecture for better maintainability, readability, and development workflow.

## Directory Structure

```
nfl/
├── app.py                      # Main application entry point (125 lines)
├── app_original.py             # Backup of original monolithic app (1311 lines)
├── main.py                     # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Main project documentation
├── PROJECT_STRUCTURE.md        # This file
├── 
├── src/                        # Core ML modules
│   ├── model.py               # ML model implementation
│   └── data_processing.py     # Data loading and feature engineering
├── 
├── models/                     # Trained model storage
│   └── expected_yards_model.pkl
├── 
├── page_modules/               # Individual page modules (renamed from pages/)
│   ├── play_predictor.py      # AI play recommendation page
│   ├── analytics_dashboard.py # Data analytics and visualizations
│   ├── model_insights.py      # Model explainability features
│   ├── scenario_simulator.py  # Game scenario testing
│   ├── data_explorer.py       # Interactive data exploration
│   └── player_matchup_analyzer.py  # Player-vs-player matchup analytics (NEW)
├── 
└── utils/                      # Utility modules
    ├── styles.py              # CSS styling and themes
    └── data_utils.py          # Data loading and management
```

## Module Breakdown

### Main Application (`app.py`) - 125 lines
- **Purpose**: Clean entry point and routing
- **Functions**:
  - Page configuration and styling application
  - Header and sidebar rendering
  - Navigation routing to appropriate page modules
  - Error handling and model status checking

### Page Modules (`page_modules/`)

#### 1. Play Predictor (`play_predictor.py`)
- AI-powered play recommendations
- Interactive game situation inputs
- Visual recommendation displays
- Contextual alerts and advice

#### 2. Analytics Dashboard (`analytics_dashboard.py`) 
- Key performance indicators
- Performance analysis charts
- Field position impact analysis
- Success rate visualizations

#### 3. Model Insights (`model_insights.py`)
- Feature importance analysis
- Interactive prediction explanations
- Model interpretability features using built-in feature importance
- Professional data visualizations

#### 4. Scenario Simulator (`scenario_simulator.py`)
- Predefined critical game scenarios
- Custom scenario builder
- Gauge visualizations for advantages
- Strategic analysis and warnings

#### 5. Data Explorer (`data_explorer.py`)
- Interactive data filtering
- Distribution analysis
- Correlation matrices
- Raw data viewing and download

#### 6. Player Matchup Analyzer (`player_matchup_analyzer.py`)
- Analyze individual player matchups (e.g., WR vs CB, OL vs DL)
- Filter by situation, formation, personnel, and more
- Visualize matchup win rates and expected outcomes
- Data always uses the 5 most recent NFL seasons

### Utility Modules (`utils/`)

#### 1. Styles (`styles.py`) - 269 lines
- Complete CSS styling system
- Professional color schemes
- Accessibility improvements
- Responsive design features
- Animation and interaction effects

#### 2. Data Utils (`data_utils.py`) - 81 lines
- Model loading and caching
- Setup instructions display
- Error handling for data operations

## Usage

### Running the Application
```bash
python -m streamlit run app.py
```

### Development
1. **Adding New Pages**: Create new modules in `page_modules/` directory
2. **Styling Changes**: Modify `utils/styles.py`
3. **Utility Functions**: Add to `utils/data_utils.py`
4. **Import in Main**: Add imports and routing in `app.py`

## Import Structure

The modular design uses clean imports with implicit namespace packages:
```python
# Utilities
from utils.styles import apply_custom_styles
from utils.data_utils import load_model, display_setup_instructions

# Page modules
from page_modules.play_predictor import play_predictor_page
from page_modules.analytics_dashboard import analytics_dashboard_page
# ... etc
```