import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
from pathlib import Path
import os
import warnings
import datetime
import urllib3
import ssl
import certifi
from urllib.request import urlopen
warnings.filterwarnings('ignore')

def download_nfl_data(years=None, max_samples_per_year=15000):
    """
    Download NFL play-by-play data from nflverse-data repository
    
    Args:
        years: List of years to download (default: 5 most recent seasons)
        max_samples_per_year: Maximum number of plays to sample per year for efficiency
    
    Returns:
        DataFrame: Combined NFL play-by-play data
    """
    print("🏈 Downloading NFL play-by-play data from nflverse-data...")
    
    # Correct base URL for direct downloads from the pbp release
    base_url = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
    
    all_dfs = []
    
    if years is None:
        current_year = datetime.datetime.now().year
        # NFL season is usually completed by February, so if before March, use previous year as last completed
        if datetime.datetime.now().month < 3:
            current_year -= 1
        years = [current_year - i for i in range(5)]
    
    print(f"Attempting to download data for years: {years}")
    
    for year in years:
        urls_to_try = [
            f"{base_url}/play_by_play_{year}.parquet",
            f"{base_url}/play_by_play_{year}.csv.gz"
        ]
        for url in urls_to_try:
            try:
                print(f"  📥 Trying URL: {url}")
                response = requests.get(url, verify=certifi.where())
                response.raise_for_status()
                if response.headers.get('content-type', '').startswith('text/html'):
                    print(f"  ⚠️ Received HTML instead of data from {url}")
                    continue
                import io
                if url.endswith('.parquet'):
                    df = pd.read_parquet(io.BytesIO(response.content))
                elif url.endswith('.csv.gz'):
                    df = pd.read_csv(io.BytesIO(response.content), compression='gzip', low_memory=False)
                else:
                    df = pd.read_csv(io.BytesIO(response.content), low_memory=False)
                print(f"  ✅ Downloaded {len(df):,} plays from {year}")
                print(f"  Filtering plays for {year}...")
                df = df[
                    (df['play_type'].isin(['run', 'pass'])) &
                    (df['yards_gained'].notna()) &
                    (df['down'].notna()) &
                    (df['ydstogo'].notna()) &
                    (df['yardline_100'].notna())
                ].copy()
                if len(df) > max_samples_per_year:
                    print(f"  Sampling {max_samples_per_year:,} plays from {len(df):,} total plays")
                    df = df.sample(n=max_samples_per_year, random_state=42)
                all_dfs.append(df)
                print(f"  ✅ Filtered to {len(df):,} relevant plays")
                break
            except requests.exceptions.SSLError as e:
                print(f"  ⚠️ SSL Error for {url}: {str(e)}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"  ⚠️ Request failed for {url}: {str(e)}")
                continue
            except Exception as e:
                print(f"  ⚠️ Failed to download from {url}: {str(e)}")
                continue
    if all_dfs:
        print("Combining data from all years...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"🎯 Total NFL plays loaded: {len(combined_df):,}")
        return combined_df
    print("⚠️ Failed to download from primary sources, trying alternative method...")
    try:
        fallback_url = f"{base_url}/play_by_play_2020.csv.gz"
        print(f"Attempting fallback URL: {fallback_url}")
        response = requests.get(fallback_url, verify=certifi.where())
        response.raise_for_status()
        if response.headers.get('content-type', '').startswith('text/html'):
            raise Exception("Received HTML instead of data file")
        import io
        df = pd.read_csv(io.BytesIO(response.content), compression='gzip', low_memory=False)
        print("Filtering fallback data...")
        df = df[
            (df['play_type'].isin(['run', 'pass'])) &
            (df['yards_gained'].notna()) &
            (df['down'].notna()) &
            (df['ydstogo'].notna()) &
            (df['yardline_100'].notna())
        ].copy()
        if len(df) > max_samples_per_year:
            print(f"Sampling {max_samples_per_year:,} plays from fallback data")
            df = df.sample(n=max_samples_per_year, random_state=42)
        print(f"✅ Downloaded {len(df):,} plays from fallback source")
        return df
    except Exception as e:
        print(f"⚠️ Fallback also failed: {e}")
        print(f"Error details: {str(e)}")
        raise Exception(f"❌ Failed to download data from any source. Error: {str(e)}")

def load_nfl_data():
    """
    Main function to load NFL data from nflverse-data
    """
    try:
        print("Starting NFL data loading process...")
        # Always use 5 most recent completed seasons
        df = download_nfl_data(years=None, max_samples_per_year=8000)
        if df is not None and len(df) > 0:
            print(f"✅ Successfully loaded {len(df):,} real NFL plays")
            return df
        else:
            raise Exception("No data downloaded from primary source")
    
    except Exception as e:
        print(f"⚠️ Could not download NFL data: {e}")
        print(f"Error details: {str(e)}")
        raise Exception(f"❌ Failed to load NFL data. Error: {str(e)}")

def clean_and_filter_data(df):
    """
    Clean and prepare the NFL data for modeling
    """
    print("🧹 Cleaning and filtering NFL data...")
    
    initial_count = len(df)
    
    # Ensure required columns exist
    required_cols = ['play_type', 'yards_gained', 'down', 'ydstogo', 'yardline_100']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    # Filter for relevant plays only
    df = df[df['play_type'].isin(['run', 'pass'])].copy()
    
    # Remove plays with missing critical data
    df = df.dropna(subset=required_cols)
    
    # Apply realistic constraints
    df = df[
        (df['yards_gained'] >= -25) & (df['yards_gained'] <= 99) &
        (df['down'] >= 1) & (df['down'] <= 4) &
        (df['ydstogo'] >= 1) & (df['ydstogo'] <= 30) &
        (df['yardline_100'] >= 1) & (df['yardline_100'] <= 99)
    ]
    
    print(f"✅ Cleaned data: {len(df):,} plays (removed {initial_count - len(df):,})")
    return df

def engineer_comprehensive_features(df):
    """
    Create comprehensive features for ML model
    """
    print("⚙️ Engineering comprehensive features...")
    
    # Copy to avoid modifying original
    df = df.copy()
    
    # Basic play type encoding
    df['is_pass'] = (df['play_type'] == 'pass').astype(int)
    df['is_run'] = (df['play_type'] == 'run').astype(int)
    
    # Down and distance features
    df['third_down'] = (df['down'] == 3).astype(int)
    df['fourth_down'] = (df['down'] == 4).astype(int)
    df['short_yardage'] = (df['ydstogo'] <= 3).astype(int)
    df['medium_yardage'] = ((df['ydstogo'] >= 4) & (df['ydstogo'] <= 7)).astype(int)
    df['long_yardage'] = (df['ydstogo'] >= 8).astype(int)
    df['yards_per_down'] = df['ydstogo'] / df['down']
    
    # Field position features
    df['distance_to_goal'] = df['yardline_100']
    df['red_zone'] = (df['yardline_100'] <= 20).astype(int)
    df['green_zone'] = (df['yardline_100'] <= 10).astype(int)
    df['goal_line'] = (df['yardline_100'] <= 5).astype(int)
    df['midfield'] = ((df['yardline_100'] >= 40) & (df['yardline_100'] <= 60)).astype(int)
    df['own_territory'] = (df['yardline_100'] >= 50).astype(int)
    
    # Game context features
    if 'score_differential' in df.columns:
        df['score_diff'] = df['score_differential'].fillna(0)
    else:
        df['score_diff'] = 0
    
    df['losing'] = (df['score_diff'] < 0).astype(int)
    df['winning'] = (df['score_diff'] > 0).astype(int)
    df['tied'] = (df['score_diff'] == 0).astype(int)
    df['close_game'] = (abs(df['score_diff']) <= 7).astype(int)
    df['blowout'] = (abs(df['score_diff']) > 14).astype(int)
    
    # Time context
    if 'quarter' in df.columns:
        df['quarter'] = df['quarter'].fillna(1)
    else:
        df['quarter'] = np.random.choice([1, 2, 3, 4], len(df))
    
    df['first_quarter'] = (df['quarter'] == 1).astype(int)
    df['second_quarter'] = (df['quarter'] == 2).astype(int)
    df['third_quarter'] = (df['quarter'] == 3).astype(int)
    df['fourth_quarter'] = (df['quarter'] == 4).astype(int)
    df['first_half'] = (df['quarter'] <= 2).astype(int)
    df['second_half'] = (df['quarter'] >= 3).astype(int)
    
    # Situational features
    df['passing_down'] = ((df['down'] == 2) & (df['ydstogo'] >= 8) | 
                         (df['down'] == 3) & (df['ydstogo'] >= 5)).astype(int)
    df['running_down'] = ((df['down'] <= 2) & (df['ydstogo'] <= 4)).astype(int)
    df['obvious_pass'] = ((df['down'] == 3) & (df['ydstogo'] >= 10)).astype(int)
    df['obvious_run'] = ((df['ydstogo'] <= 2) & (df['down'] <= 3)).astype(int)
    
    # Advanced situational combinations
    df['third_and_long'] = ((df['down'] == 3) & (df['ydstogo'] >= 7)).astype(int)
    df['third_and_short'] = ((df['down'] == 3) & (df['ydstogo'] <= 3)).astype(int)
    df['fourth_and_short'] = ((df['down'] == 4) & (df['ydstogo'] <= 2)).astype(int)
    df['red_zone_third_down'] = (df['red_zone'] & df['third_down']).astype(int)
    df['goal_line_situation'] = (df['goal_line'] & (df['ydstogo'] >= df['yardline_100'])).astype(int)
    
    print(f"✅ Feature engineering complete: {df.shape[1]} total features")
    return df

def prepare_model_data(df):
    """
    Prepare final feature set and target for model training
    """
    print("🎯 Preparing data for model training...")
    
    # Define feature columns (excluding target and identifiers)
    feature_columns = [
        # Basic situation
        'down', 'ydstogo', 'distance_to_goal', 'is_pass', 'is_run',
        
        # Down situation
        'third_down', 'fourth_down', 'short_yardage', 'medium_yardage', 'long_yardage',
        'yards_per_down',
        
        # Field position
        'red_zone', 'green_zone', 'goal_line', 'midfield', 'own_territory',
        
        # Game context
        'score_diff', 'losing', 'winning', 'tied', 'close_game', 'blowout',
        
        # Time context
        'first_quarter', 'second_quarter', 'third_quarter', 'fourth_quarter',
        'first_half', 'second_half',
        
        # Situational
        'passing_down', 'running_down', 'obvious_pass', 'obvious_run',
        'third_and_long', 'third_and_short', 'fourth_and_short',
        'red_zone_third_down', 'goal_line_situation'
    ]
    
    # Only include features that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['yards_gained']
    
    print(f"✅ Prepared {len(available_features)} features for {len(X):,} samples")
    print(f"   Target variable (yards_gained): mean={y.mean():.2f}, std={y.std():.2f}")
    
    return X, y, available_features

def get_play_features(down, ydstogo, yardline_100, quarter=1, score_diff=0):
    """
    Generate feature dictionary for a single play scenario
    """
    features = {
        # Basic situation
        'down': down,
        'ydstogo': ydstogo,
        'distance_to_goal': yardline_100,
        
        # Down situation
        'third_down': 1 if down == 3 else 0,
        'fourth_down': 1 if down == 4 else 0,
        'short_yardage': 1 if ydstogo <= 3 else 0,
        'medium_yardage': 1 if 4 <= ydstogo <= 7 else 0,
        'long_yardage': 1 if ydstogo >= 8 else 0,
        'yards_per_down': ydstogo / down,
        
        # Field position
        'red_zone': 1 if yardline_100 <= 20 else 0,
        'green_zone': 1 if yardline_100 <= 10 else 0,
        'goal_line': 1 if yardline_100 <= 5 else 0,
        'midfield': 1 if 40 <= yardline_100 <= 60 else 0,
        'own_territory': 1 if yardline_100 >= 50 else 0,
        
        # Game context
        'score_diff': score_diff,
        'losing': 1 if score_diff < 0 else 0,
        'winning': 1 if score_diff > 0 else 0,
        'tied': 1 if score_diff == 0 else 0,
        'close_game': 1 if abs(score_diff) <= 7 else 0,
        'blowout': 1 if abs(score_diff) > 14 else 0,
        
        # Time context
        'first_quarter': 1 if quarter == 1 else 0,
        'second_quarter': 1 if quarter == 2 else 0,
        'third_quarter': 1 if quarter == 3 else 0,
        'fourth_quarter': 1 if quarter == 4 else 0,
        'first_half': 1 if quarter <= 2 else 0,
        'second_half': 1 if quarter >= 3 else 0,
        
        # Situational
        'passing_down': 1 if (down == 2 and ydstogo >= 8) or (down == 3 and ydstogo >= 5) else 0,
        'running_down': 1 if down <= 2 and ydstogo <= 4 else 0,
        'obvious_pass': 1 if down == 3 and ydstogo >= 10 else 0,
        'obvious_run': 1 if ydstogo <= 2 and down <= 3 else 0,
        'third_and_long': 1 if down == 3 and ydstogo >= 7 else 0,
        'third_and_short': 1 if down == 3 and ydstogo <= 3 else 0,
        'fourth_and_short': 1 if down == 4 and ydstogo <= 2 else 0,
        'red_zone_third_down': 1 if yardline_100 <= 20 and down == 3 else 0,
        'goal_line_situation': 1 if yardline_100 <= 5 and ydstogo >= yardline_100 else 0
    }
    
    return features

# Main processing pipeline
def load_and_prepare_data():
    """
    Complete data loading and preparation pipeline
    """
    print("🏈 Starting NFL data processing pipeline...")
    print("=" * 60)
    
    df = load_nfl_data()
    
    df = clean_and_filter_data(df)
    if df is None:
        raise Exception("Data cleaning failed")
    
    df = engineer_comprehensive_features(df)
    
    X, y, feature_names = prepare_model_data(df)
    
    print("=" * 60)
    print("✅ Data processing pipeline complete!")
    
    return X, y, feature_names, df
