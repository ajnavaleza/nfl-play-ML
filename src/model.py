import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import shap
from math import sqrt

class ExpectedYardsModel:    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.explainer = None
        self.is_trained = False
        
    def train_model(self, X, y, feature_names=None):
        self.feature_names = feature_names if feature_names else list(X.columns)
        
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # initialize model based on type
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror'
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        # train the model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # evaluate the model
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        # calculate metrics
        train_rmse = sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = sqrt(mean_squared_error(y_test, test_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        print(f"\n=== Model Performance ===")
        print(f"Train RMSE: {train_rmse:.3f} yards")
        print(f"Test RMSE:  {test_rmse:.3f} yards")
        print(f"Train MAE:  {train_mae:.3f} yards")
        print(f"Test MAE:   {test_mae:.3f} yards")
        print(f"Train R²:   {train_r2:.3f}")
        print(f"Test R²:    {test_r2:.3f}")
        
        # initialize SHAP explainer
        print("\ninitializing SHAP explainer...")
        if self.model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        
        self.is_trained = True
        return self.model
    
    def predict_expected_yards(self, features_dict, play_type='pass'):
        # predict expected yards for a given situation
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # convert features dict to dataframe
        features_df = pd.DataFrame([features_dict])
        
        # add play type feature
        features_df['is_pass'] = 1 if play_type == 'pass' else 0
        features_df['is_run'] = 1 if play_type == 'run' else 0
        
        # ensure all required features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # select only the features used in training
        features_df = features_df[self.feature_names]
        
        # make prediction
        prediction = self.model.predict(features_df)[0]
        return max(0, prediction)
    
    def recommend_play_type(self, features_dict):
        # recommend optimal play type based on expected yards
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # predict expected yards for both run and pass
        run_expected = self.predict_expected_yards(features_dict, 'run')
        pass_expected = self.predict_expected_yards(features_dict, 'pass')
        
        # create recommendation
        recommendation = {
            'run_expected_yards': run_expected,
            'pass_expected_yards': pass_expected,
            'recommended_play': 'pass' if pass_expected > run_expected else 'run',
            'expected_yards_difference': abs(pass_expected - run_expected),
            'confidence': 'high' if abs(pass_expected - run_expected) > 1.0 else 'moderate'
        }
        
        # add contextual advice
        down = features_dict.get('down', 1)
        ydstogo = features_dict.get('ydstogo', 10)
        yardline = features_dict.get('distance_to_goal', 50)
        
        if down == 4 and ydstogo <= 2:
            recommendation['context_advice'] = "Short yardage situation - consider power run"
        elif down == 3 and ydstogo >= 8:
            recommendation['context_advice'] = "Passing down - defense expects pass"
        elif yardline <= 10:
            recommendation['context_advice'] = "Red zone - compressed field affects passing"
        else:
            recommendation['context_advice'] = "Standard down - use expected yards as guide"
        
        return recommendation
    
    def explain_prediction(self, features_dict, play_type='pass'):
        # get SHAP explanation for a prediction
        if not self.is_trained or self.explainer is None:
            raise ValueError("Model and explainer must be initialized")
        
        # convert features dict to dataframe
        features_df = pd.DataFrame([features_dict])
        
        # add play type feature
        features_df['is_pass'] = 1 if play_type == 'pass' else 0
        features_df['is_run'] = 1 if play_type == 'run' else 0
        
        # ensure all required features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # select only the features used in training
        features_df = features_df[self.feature_names]
        
        # get SHAP values
        shap_values = self.explainer.shap_values(features_df)
        
        # create explanation dictionary
        explanation = {}
        for i, feature in enumerate(self.feature_names):
            explanation[feature] = {
                'value': features_df[feature].iloc[0],
                'shap_value': shap_values[0][i],
                'contribution': 'positive' if shap_values[0][i] > 0 else 'negative'
            }
        
        # sort by absolute SHAP value
        sorted_explanation = dict(sorted(
            explanation.items(), 
            key=lambda x: abs(x[1]['shap_value']), 
            reverse=True
        ))
        
        return sorted_explanation
    
    def get_feature_importance(self):
        # get feature importance from the trained model
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def save_model(self, filepath):
        # save the trained model
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        # load a trained model
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            # reinitialize explainer
            if self.is_trained:
                if self.model_type == 'xgboost':
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

# Legacy function for backwards compatibility
def train_model(X, y):
    """legacy function - use ExpectedYardsModel class instead"""
    model = ExpectedYardsModel()
    return model.train_model(X, y)
