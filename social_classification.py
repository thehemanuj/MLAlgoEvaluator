import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class Classification:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.algo = {
            'rfc': RandomForestClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svc': SVC(random_state=42),
            'dtc': DecisionTreeClassifier(random_state=42),
            'gnb': GaussianNB(),
            'gbc': GradientBoostingClassifier(random_state=42),
            'abc': AdaBoostClassifier(random_state=42),
            'etc': ExtraTreesClassifier(random_state=42),
            'qda': QuadraticDiscriminantAnalysis(),
            'mlp': MLPClassifier(max_iter=1000, random_state=42),
            'ridge': RidgeClassifier(random_state=42)
        }
        
        self.trained_models = {}
        self.best_model_name = None
        self.best_model = None
        self.results = {}
    
    def train(self, key='best'):
        if key == 'best':
            # Train all models
            print("Training all models...")
            for model_key in self.algo.keys():
                print(f"Training {model_key}...")
                self.algo[model_key].fit(self.X_train, self.y_train)
                self.trained_models[model_key] = self.algo[model_key]
            
            self.results = self.evaluate_all()
            self.best_model_name, best_metrics = self.select_best_model()
            self.best_model = self.trained_models[self.best_model_name]
            
            print(f"\nBest Model: {self.best_model_name}")
            print(f"Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"Precision: {best_metrics['precision']:.4f}")
            print(f"Recall: {best_metrics['recall']:.4f}")
            print(f"F1 Score: {best_metrics['f1']:.4f}")
            
            return {
                'best_model': self.best_model_name,
                'metrics': best_metrics,
                'all_results': self.results
            }
        else:
            # Train specific model
            if key not in self.algo:
                raise ValueError(f"Invalid model key. Choose from: {list(self.algo.keys())}")
            
            print(f"Training {key}...")
            self.algo[key].fit(self.X_train, self.y_train)
            self.trained_models[key] = self.algo[key]
            self.best_model = self.algo[key]
            self.best_model_name = key
            
            # Evaluate single model
            metrics = self.evaluate_single(key)
            self.results[key] = metrics
            
            print(f"\nModel: {key}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
            return {
                'model': key,
                'metrics': metrics
            }
    
    def evaluate_single(self, model_key):
        """Evaluate a single trained model."""
        model = self.trained_models[model_key]
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(self.y_train))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=average, zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def evaluate_all(self):
        """Evaluate all trained models."""
        results = {}
        
        for model_key in self.trained_models.keys():
            results[model_key] = self.evaluate_single(model_key)
        
        return results
    
    def select_best_model(self):
        best_model = None
        best_f1 = -float('inf')
        best_accuracy = -float('inf')
        
        for model_key, metrics in self.results.items():
            f1 = metrics['f1']
            accuracy = metrics['accuracy']
            
            # Primary criterion: highest F1 score
            # Secondary criterion: highest accuracy (if F1 is close)
            if f1 > best_f1 or (abs(f1 - best_f1) < 0.01 and accuracy > best_accuracy):
                best_model = model_key
                best_f1 = f1
                best_accuracy = accuracy
        
        return best_model, self.results[best_model]
    
    def predict(self, X_new):
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        return self.best_model.predict(X_new)
    
    def predict_proba(self, X_new):
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError(f"Model {self.best_model_name} does not support probability predictions.")
        
        return self.best_model.predict_proba(X_new)
    
    def get_model_comparison(self):
        if not self.results:
            raise ValueError("No models evaluated yet. Call train() first.")
        
        comparison = []
        for model_key, metrics in self.results.items():
            comparison.append({
                'Model': model_key,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_classification_report(self, model_key=None):
        """
        Get detailed classification report for a specific model or the best model.
        
        Parameters:
        -----------
        model_key : str, optional
            Specific model to get report for. If None, uses best model.
        
        Returns:
        --------
        str : Classification report
        """
        if model_key is None:
            if self.best_model is None:
                raise ValueError("No model trained yet. Call train() first.")
            model_key = self.best_model_name
        
        if model_key not in self.results:
            raise ValueError(f"Model {model_key} has not been evaluated yet.")
        
        y_pred = self.results[model_key]['predictions']
        
        return classification_report(self.y_test, y_pred)
    
    def get_confusion_matrix(self, model_key=None):
        """
        Get confusion matrix for a specific model or the best model.
        
        Parameters:
        -----------
        model_key : str, optional
            Specific model to get matrix for. If None, uses best model.
        
        Returns:
        --------
        array : Confusion matrix
        """
        if model_key is None:
            if self.best_model is None:
                raise ValueError("No model trained yet. Call train() first.")
            model_key = self.best_model_name
        
        if model_key not in self.results:
            raise ValueError(f"Model {model_key} has not been evaluated yet.")
        
        return self.results[model_key]['confusion_matrix']