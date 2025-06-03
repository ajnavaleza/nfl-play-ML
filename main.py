from src.data_processing import load_data, clean_data, feature_engineering, prepare_features_and_target
from src.model import ExpectedYardsModel

def main():
    print("🏈 Expected Yards & Play Calling Intelligence System")
    print("=" * 60)
    
    df = load_data('data/nfl_pbp_2021_2022.csv')
    print(f"Loaded {len(df):,} plays")
    
    df = clean_data(df)    
    df = feature_engineering(df)
    print("Feature engineering completed")
    
    X, y, feature_names = prepare_features_and_target(df)
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {len(X):,}")
    
    model = ExpectedYardsModel(model_type='xgboost')
    trained_model = model.train_model(X, y, feature_names)
    print(f"EXPY model trained")
    
    model.save_model('models/expected_yards_model.pkl')
    print(f"EXPY model saved")
    
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    main()
