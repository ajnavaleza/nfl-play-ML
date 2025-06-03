# 🏈 Expected Yards & Play Calling Intelligence System

An ML-powered web application that predicts expected yards gained (xY) for various play types and recommends optimal offensive plays based on down, distance, field position, and defense tendencies.

## Description

This system can help offensive coordinators and analysts evaluate play-calling effectiveness, simulate decisions in critical game scenarios, and create data-informed game plans tailored to specific opponents.

## Features

- **Expected Yards Prediction**: ML model predicting yards gained based on game context
- **Play Recommendation System**: Intelligent suggestions for optimal play types
- **Interactive Dashboard**: Real-time simulation and visualization
- **Opponent Analysis**: Customizable analysis based on defensive tendencies
- **Model Explainability**: SHAP values showing key factors influencing predictions
- **Game Scenario Simulator**: Test different play calling strategies

## 🛠️ Tech Stack

- **Backend**: Python, XGBoost, scikit-learn
- **Frontend**: Streamlit
- **Visualization**: Plotly, Seaborn
- **Data**: NFL play-by-play data (nflfastR format)
- **ML Explainability**: SHAP

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model**:
   ```bash
   python main.py
   ```

2. **Launch the web app**:
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard** at `http://localhost:8501`

## Data

The system uses NFL play-by-play data including:
- Down and distance
- Field position
- Formation and personnel
- Play type and outcome
- Game context (score, time, etc.)

## Model Features

- **Gradient Boosting (XGBoost)** for expected yards prediction
- **Feature Engineering** including formation, motion, personnel groupings
- **Contextual Recommendations** based on game situation
- **Model Explainability** via SHAP values

## Dashboard Features

- **Play Predictor**: Input game situation and get expected yards
- **Play Recommender**: Get optimal play suggestions
- **Scenario Simulator**: Simulate full drives and outcomes
- **Analytics Dashboard**: Visualize trends and patterns
- **Model Insights**: Understand what drives predictions
