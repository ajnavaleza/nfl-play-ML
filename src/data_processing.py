import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Load NFL play-by-play data"""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean and filter the data for modeling"""
    # Drop rows with missing target or critical features
    critical_cols = ['yards_gained', 'down', 'ydstogo', 'yardline_100', 'play_type']
    df = df.dropna(subset=critical_cols)
    
    # Filter relevant plays (run/pass only, exclude special teams)
    df = df[df['play_type'].isin(['run', 'pass'])]
    
    # Remove obvious outliers and errors
    df = df[(df['yards_gained'] >= -30) & (df['yards_gained'] <= 99)]
    df = df[(df['ydstogo'] > 0) & (df['ydstogo'] <= 30)]
    df = df[(df['down'] >= 1) & (df['down'] <= 4)]
    
    return df

def feature_engineering(df):
    """Create comprehensive features for the ML model"""
    
    # Basic play information
    df['is_pass'] = (df['play_type'] == 'pass').astype(int)
    df['is_run'] = (df['play_type'] == 'run').astype(int)
    
    # Field position features
    df['distance_to_goal'] = df['yardline_100']
    df['field_side'] = np.where(df['yardline_100'] > 50, 'own', 'opponent')
    df['red_zone'] = (df['yardline_100'] <= 20).astype(int)
    df['goal_line'] = (df['yardline_100'] <= 5).astype(int)
    
    # Down and distance features
    df['down'] = df['down'].astype(int)
    df['third_down'] = (df['down'] == 3).astype(int)
    df['fourth_down'] = (df['down'] == 4).astype(int)
    df['short_yardage'] = (df['ydstogo'] <= 3).astype(int)
    df['long_yardage'] = (df['ydstogo'] >= 8).astype(int)
    df['yards_per_down'] = df['ydstogo'] / df['down']
    
    # Game context features (if available)
    if 'score_differential' in df.columns:
        df['score_diff'] = df['score_differential']
        df['losing'] = (df['score_differential'] < 0).astype(int)
        df['winning_big'] = (df['score_differential'] > 14).astype(int)
    else:
        df['score_diff'] = 0
        df['losing'] = 0
        df['winning_big'] = 0
    
    # Time features (if available)
    if 'quarter' in df.columns:
        df['quarter'] = df['quarter'].fillna(1)
        df['fourth_quarter'] = (df['quarter'] == 4).astype(int)
        df['first_half'] = (df['quarter'] <= 2).astype(int)
    else:
        df['quarter'] = 1
        df['fourth_quarter'] = 0
        df['first_half'] = 1
    
    # Formation features (if available)
    if 'formation' in df.columns:
        df['formation'] = df['formation'].fillna('unknown')
        # Create dummy variables for common formations
        formation_dummies = pd.get_dummies(df['formation'], prefix='formation')
        df = pd.concat([df, formation_dummies], axis=1)
    
    # Personnel features (if available)
    if 'personnel_offense' in df.columns:
        df['personnel_offense'] = df['personnel_offense'].fillna('11_personnel')
        personnel_dummies = pd.get_dummies(df['personnel_offense'], prefix='personnel')
        df = pd.concat([df, personnel_dummies], axis=1)
    
    # Historical success features
    df['expected_yards'] = 0  # Placeholder for iterative calculation
    
    return df

def prepare_features_and_target(df):
    """Prepare features and target for model training"""
    
    # Core features that should always be available
    base_features = [
        'down', 'ydstogo', 'distance_to_goal', 'is_pass', 'is_run',
        'red_zone', 'goal_line', 'third_down', 'fourth_down',
        'short_yardage', 'long_yardage', 'yards_per_down',
        'score_diff', 'losing', 'winning_big', 'fourth_quarter', 'first_half'
    ]
    
    # Add formation features if they exist
    formation_cols = [col for col in df.columns if col.startswith('formation_')]
    personnel_cols = [col for col in df.columns if col.startswith('personnel_')]
    
    feature_cols = base_features + formation_cols + personnel_cols
    
    # Only include features that actually exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0)  # Fill any remaining NaNs with 0
    y = df['yards_gained']
    
    return X, y, feature_cols

def get_play_context_features(down, ydstogo, yardline_100, quarter=1, score_diff=0):
    """Create features for a single play prediction"""
    features = {
        'down': down,
        'ydstogo': ydstogo,
        'distance_to_goal': yardline_100,
        'red_zone': 1 if yardline_100 <= 20 else 0,
        'goal_line': 1 if yardline_100 <= 5 else 0,
        'third_down': 1 if down == 3 else 0,
        'fourth_down': 1 if down == 4 else 0,
        'short_yardage': 1 if ydstogo <= 3 else 0,
        'long_yardage': 1 if ydstogo >= 8 else 0,
        'yards_per_down': ydstogo / down,
        'score_diff': score_diff,
        'losing': 1 if score_diff < 0 else 0,
        'winning_big': 1 if score_diff > 14 else 0,
        'fourth_quarter': 1 if quarter == 4 else 0,
        'first_half': 1 if quarter <= 2 else 0
    }
    
    return features
