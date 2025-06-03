from src.data_processing import load_and_prepare_data, get_play_features
from src.model import ExpectedYardsModel
import os

def main():
    print("🏈 Expected Yards & Play Calling Intelligence System")
    print("📊 Powered by Real NFL Data from nflfastR")
    print("=" * 70)
    
    try:
        # load and prepare all data using the streamlined pipeline
        print("\nLoading and preparing NFL data...")
        X, y, feature_names, raw_df = load_and_prepare_data()
        
        print(f"\nDataset Summary:")
        print(f"   • Total plays: {len(X):,}")
        print(f"   • Features: {len(feature_names)}")
        print(f"   • Average yards/play: {y.mean():.2f}")
        print(f"   • Pass plays: {(raw_df['play_type'] == 'pass').sum():,} ({(raw_df['play_type'] == 'pass').mean():.1%})")
        print(f"   • Run plays: {(raw_df['play_type'] == 'run').sum():,} ({(raw_df['play_type'] == 'run').mean():.1%})")
        
        print("\nTraining Expected Yards Model with XGBoost...")
        model = ExpectedYardsModel(model_type='xgboost')
        trained_model = model.train_model(X, y, feature_names)
        
        print("\nSaving trained model...")
        os.makedirs('models', exist_ok=True)
        model.save_model('models/expected_yards_model.pkl')
        
        print("\nTop 15 Most Important Features:")
        importance = model.get_feature_importance()
        for i, (feature, score) in enumerate(list(importance.items())[:15]):
            print(f"{i+1:2d}. {feature:<25} {score:.4f}")
        
        print("\nTesting Model Predictions...")
        print("-" * 50)
        
        test_scenarios = [
            {
                'name': "3rd & 7 at midfield",
                'params': {'down': 3, 'ydstogo': 7, 'yardline_100': 50, 'quarter': 2, 'score_diff': 0}
            },
            {
                'name': "1st & 10 in red zone",
                'params': {'down': 1, 'ydstogo': 10, 'yardline_100': 15, 'quarter': 3, 'score_diff': -3}
            },
            {
                'name': "4th & 2 at goal line",
                'params': {'down': 4, 'ydstogo': 2, 'yardline_100': 3, 'quarter': 4, 'score_diff': -7}
            },
            {
                'name': "2nd & 15 in own territory",
                'params': {'down': 2, 'ydstogo': 15, 'yardline_100': 75, 'quarter': 1, 'score_diff': 10}
            }
        ]
        
        for scenario in test_scenarios:
            features = get_play_features(**scenario['params'])
            recommendation = model.recommend_play_type(features)
            
            print(f"\n {scenario['name']}:")
            print(f"   Recommended: {recommendation['recommended_play'].upper()}")
            print(f"   Run expected: {recommendation['run_expected_yards']:.2f} yards")
            print(f"   Pass expected: {recommendation['pass_expected_yards']:.2f} yards")
            print(f"   Confidence: {recommendation['confidence'].title()}")
            print(f"   Strategy: {recommendation['context_advice']}")
        
        print("\n" + "=" * 70)
        print("Model Training Complete")
        print("\nData Source Information:")
        print("   • Data automatically downloaded from nflfastR public repository")
        print("   • Uses recent NFL seasons (2022-2023) for current relevance") 
        print("   • Falls back to realistic synthetic data if download fails")
        print("   • No large CSV files required!")
        
        print(f"\nNext Steps:")
        print(f"   1. Run 'streamlit run app.py' to launch the web dashboard")
        print(f"   2. Explore play recommendations and analytics")
        print(f"   3. Test different game scenarios")
        
        print(f"\nThe model uses {len(feature_names)} advanced features including:")
        print(f"   • Down, distance, and field position")
        print(f"   • Game context (score, quarter)")
        print(f"   • Situational awareness (red zone, passing down, etc.)")
        print(f"   • Advanced play type recognition")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nTroubleshooting:")
        print("   • Check internet connection for data download")
        print("   • Ensure all required packages are installed")
        print("   • Try running 'pip install -r requirements.txt'")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready !")
    else:
        print("\n⚠️ Setup incomplete. Please resolve errors above.")
