

"""
Model training and evaluation for lead scoring
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import yaml
import joblib
from datetime import datetime
from typing import Tuple, Dict
import os

class ModelTrainer:
    """Train and evaluate lead scoring models"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_model(self, algorithm: str):
        """Initialize model based on algorithm choice"""
        if algorithm == 'logistic':
            return LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=1000,
                class_weight='balanced'
            )
        elif algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                n_jobs=-1
            )
        elif algorithm == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['model']['random_state'],
                scale_pos_weight=1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif algorithm == 'lightgbm':
            return LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              use_smote: bool = True, tune_hyperparameters: bool = False) -> Dict:
        """
        Train model on provided data
        
        Args:
            X: Feature matrix
            y: Target variable
            use_smote: Whether to use SMOTE for class balancing
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        # Apply SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=self.config['model']['random_state'])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"Applied SMOTE: Training samples: {len(X_train)}")
        
        # Initialize model
        algorithm = self.config['model']['algorithm']
        self.model = self.get_model(algorithm)
        
        # Hyperparameter tuning (optional)
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(
                self.model, X_train, y_train, algorithm
            )
        
        # Train model
        print(f"Training {algorithm} model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'model_version': self.model_version,
            'algorithm': algorithm,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1_score': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_test),
            'features_used': ','.join(X.columns.tolist())
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=self.config['model']['cv_folds'], 
            scoring='f1'
        )
        self.metrics['cv_f1_mean'] = cv_scores.mean()
        self.metrics['cv_f1_std'] = cv_scores.std()
        
        # Feature importance
        self._extract_feature_importance(X.columns)
        
        # Print results
        self._print_evaluation_results(y_test, y_pred_test)
        
        return self.metrics
    
    def _tune_hyperparameters(self, model, X_train, y_train, algorithm):
        """Perform grid search for hyperparameter tuning"""
        param_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'logistic': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        }
        
        if algorithm in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[algorithm],
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def _extract_feature_importance(self, feature_names):
        """Extract feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def _print_evaluation_results(self, y_test, y_pred_test):
        """Print detailed evaluation results"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"\nModel Version: {self.model_version}")
        print(f"Algorithm: {self.metrics['algorithm']}")
        print(f"\nAccuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        print(f"\nCV F1 Score: {self.metrics['cv_f1_mean']:.4f} (+/- {self.metrics['cv_f1_std']:.4f})")
        
        print("\n" + "-"*50)
        print("Classification Report:")
        print("-"*50)
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['Not Converted', 'Converted']))
        
        print("\n" + "-"*50)
        print("Confusion Matrix:")
        print("-"*50)
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"True Negatives:  {cm[0, 0]}")
        print(f"False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}")
        print(f"True Positives:  {cm[1, 1]}")
        
        print("\n" + "-"*50)
        print("Top 10 Important Features:")
        print("-"*50)
        print(self.feature_importance.head(10).to_string(index=False))
        print("="*50 + "\n")
    
    def save_model(self, path='models/lead_scorer.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'model_version': self.model_version,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path='models/lead_scorer.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data['metrics']
        self.model_version = model_data['model_version']
        print(f"✓ Model loaded from {path}")
        print(f"  Version: {self.model_version}")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")