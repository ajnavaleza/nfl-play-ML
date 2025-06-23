from src.data_processing import load_and_prepare_data, get_play_features
from src.model import ExpectedYardsModel
import os

def main():
    try:
        # load and prepare all data using the streamlined pipeline
        X, y, feature_names, raw_df = load_and_prepare_data()
        
        model = ExpectedYardsModel(model_type='xgboost')
        trained_model = model.train_model(X, y, feature_names)
        
        model.save_model('models/expected_yards_model.pkl')
        
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
        
    except Exception as e:
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready !")
    else:
        print("\n⚠️ Setup incomplete. Please resolve errors above.")
