import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


class Regression:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.algo = {
            'linreg': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'el': ElasticNet(),
            'br': BayesianRidge(),
            'svr': SVR(),
            'dtr': DecisionTreeRegressor(random_state=42),
            'rfr': RandomForestRegressor(random_state=42)
        }
        
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None
        self.results = {}
    
    def train(self, key='best'):
        if key == 'best':
            print("Training all models...")
            for model_key in self.algo.keys():
                print(f"Training {model_key}...")
                self.algo[model_key].fit(self.X_train, self.y_train)
                self.trained_models[model_key] = self.algo[model_key]
            
            # Evaluate all and find best
            self.results = self.evaluate_all()
            self.best_model_name, best_metrics = self.select_best_model()
            self.best_model = self.trained_models[self.best_model_name] 
            return {
                'best_model': self.best_model_name,
                'metrics': best_metrics,
                'all_results': self.results
            }
        else:
            # Train specific model
            if key not in self.algo:
                raise ValueError(f"Invalid model key. Choose from: {list(self.algo.keys())}")
            self.algo[key].fit(self.X_train, self.y_train)
            self.trained_models[key] = self.algo[key]
            self.best_model = self.algo[key]
            self.best_model_name = key
            metrics = self.evaluate_single(key)
            self.results[key] = metrics
            return {
                'model': key,
                'metrics': metrics
            }
    
    def evaluate_single(self, model_key):
        model = self.trained_models[model_key]
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
    
    def evaluate_all(self):
        results = {}
        
        for model_key in self.trained_models.keys():
            results[model_key] = self.evaluate_single(model_key)
        
        return results
    
    def select_best_model(self):
        best_model = None
        best_r2 = -float('inf')
        best_mse = float('inf')
        
        for model_key, metrics in self.results.items():
            r2 = metrics['r2']
            mse = metrics['mse']
            if r2 > best_r2 or (abs(r2 - best_r2) < 0.01 and mse < best_mse):
                best_model = model_key
                best_r2 = r2
                best_mse = mse
        
        return best_model, self.results[best_model]
    
    def predict(self, X_new):
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        return self.best_model.predict(X_new)
    
    def get_model_comparison(self):
        if not self.results:
            raise ValueError("No models evaluated yet. Call train() first.")
        
        comparison = []
        for model_key, metrics in self.results.items():
            comparison.append({
                'Model': model_key,
                'R² Score': metrics['r2'],
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('R² Score', ascending=False).reset_index(drop=True)
        
        return df