from src.data_processing import load_data, clean_data, feature_engineering, prepare_features_and_target
from src.model import ExpectedYardsModel

def main():
    print("🏈 Expected Yards & Play Calling Intelligence System")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    df = load_data('data/nfl_pbp_2021_2022.csv')
    print(f"Loaded {len(df):,} plays")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df):,} plays")
    
    df = feature_engineering(df)
    print("Feature engineering completed")
    
    X, y, feature_names = prepare_features_and_target(df)
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {len(X):,}")
    
    # Train the model
    print("\n🤖 Training Expected Yards Model...")
    model = ExpectedYardsModel(model_type='xgboost')
    trained_model = model.train_model(X, y, feature_names)
    
    # Save the model
    print("\n💾 Saving model...")
    model.save_model('models/expected_yards_model.pkl')
    
    # Show feature importance
    print("\n📈 Feature Importance (Top 10):")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"{i+1:2d}. {feature:<20} {score:.4f}")
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    test_features = {
        'down': 3,
        'ydstogo': 7,
        'distance_to_goal': 25,
        'quarter': 4,
        'score_diff': -3
    }
    
    recommendation = model.recommend_play_type(test_features)
    print(f"\nTest scenario: 3rd & 7 at opponent's 25-yard line, 4th quarter, down by 3")
    print(f"Recommended play: {recommendation['recommended_play'].upper()}")
    print(f"Expected yards (run): {recommendation['run_expected_yards']:.2f}")
    print(f"Expected yards (pass): {recommendation['pass_expected_yards']:.2f}")
    print(f"Advice: {recommendation['context_advice']}")
    
    print("\n✅ Model training complete! Run 'streamlit run app.py' to launch the web app.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    main()
