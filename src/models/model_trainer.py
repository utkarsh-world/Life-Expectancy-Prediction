"""Enhanced Model Training Module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Optional advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model trainer supporting multiple algorithms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.trained_models = {}
        self.scaler = StandardScaler()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'random_forest': {'enabled': True, 'n_estimators': 100, 'random_state': 42},
            'linear_regression': {'enabled': True},
            'svr': {'enabled': True, 'kernel': 'rbf', 'C': 1.0},
            'xgboost': {'enabled': XGBOOST_AVAILABLE, 'n_estimators': 100, 'random_state': 42},
            'lightgbm': {'enabled': LIGHTGBM_AVAILABLE, 'n_estimators': 100, 'random_state': 42}
        }

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        logger.info("🔧 Preparing data for training...")

        # Define target and features
        target_col = 'Life expectancy '
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Remove non-predictive columns
        exclude_cols = [target_col, 'Country', 'Year']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        X = data[feature_cols].copy()
        y = data[target_col].copy()

        # Handle categorical variables
        if 'Status' in X.columns:
            X = pd.get_dummies(X, columns=['Status'], prefix='Status')

        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])

        logger.info(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all enabled models."""
        logger.info("🤖 Starting model training...")
        logger.info("="*50)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        trained_models = {}
        results = {}

        # Train Random Forest
        if self.config.get('random_forest', {}).get('enabled', False):
            logger.info("🌳 Training Random Forest...")
            rf_model, rf_results = self._train_random_forest(X_train, y_train, X_test, y_test)
            trained_models['Random Forest'] = rf_model
            results['Random Forest'] = rf_results

        # Train Linear Regression
        if self.config.get('linear_regression', {}).get('enabled', False):
            logger.info("📈 Training Linear Regression...")
            lr_model, lr_results = self._train_linear_regression(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            trained_models['Linear Regression'] = lr_model
            results['Linear Regression'] = lr_results

        # Train SVR
        if self.config.get('svr', {}).get('enabled', False):
            logger.info("🎯 Training Support Vector Regression...")
            svr_model, svr_results = self._train_svr(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            trained_models['SVR'] = svr_model
            results['SVR'] = svr_results

        # Train XGBoost
        if self.config.get('xgboost', {}).get('enabled', False) and XGBOOST_AVAILABLE:
            logger.info("🚀 Training XGBoost...")
            xgb_model, xgb_results = self._train_xgboost(X_train, y_train, X_test, y_test)
            trained_models['XGBoost'] = xgb_model
            results['XGBoost'] = xgb_results

        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
        logger.info("="*50)
        logger.info(f"🏆 Best Model: {best_model_name}")
        logger.info(f"   R² Score: {results[best_model_name]['r2_score']:.4f}")
        logger.info(f"   RMSE: {results[best_model_name]['rmse']:.2f}")

        self.trained_models = trained_models
        return {
            'models': trained_models,
            'results': results,
            'best_model': best_model_name,
            'scaler': self.scaler
        }

    def _train_random_forest(self, X_train, y_train, X_test, y_test) -> Tuple[Any, Dict]:
        """Train Random Forest model."""
        config = self.config.get('random_forest', {})

        model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth'),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        results = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

        logger.info(f"   R² Score: {results['r2_score']:.4f}")
        logger.info(f"   RMSE: {results['rmse']:.2f} years")
        logger.info(f"   Cross-val: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")

        return model, results

    def _train_linear_regression(self, X_train, y_train, X_test, y_test) -> Tuple[Any, Dict]:
        """Train Linear Regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        results = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        logger.info(f"   R² Score: {results['r2_score']:.4f}")
        logger.info(f"   RMSE: {results['rmse']:.2f} years")

        return model, results

    def _train_svr(self, X_train, y_train, X_test, y_test) -> Tuple[Any, Dict]:
        """Train SVR model."""
        config = self.config.get('svr', {})

        model = SVR(
            kernel=config.get('kernel', 'rbf'),
            C=config.get('C', 1.0),
            gamma=config.get('gamma', 'scale'),
            epsilon=config.get('epsilon', 0.1)
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

        logger.info(f"   R² Score: {results['r2_score']:.4f}")
        logger.info(f"   RMSE: {results['rmse']:.2f} years")

        return model, results

    def _train_xgboost(self, X_train, y_train, X_test, y_test) -> Tuple[Any, Dict]:
        """Train XGBoost model."""
        config = self.config.get('xgboost', {})

        model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            random_state=config.get('random_state', 42)
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

        logger.info(f"   R² Score: {results['r2_score']:.4f}")
        logger.info(f"   RMSE: {results['rmse']:.2f} years")

        return model, results

    def save_models(self, models: Dict[str, Any], model_dir: str) -> Dict[str, str]:
        """Save trained models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_paths = {}

        for name, model in models.items():
            model_path = model_dir / f"{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
            model_paths[name] = str(model_path)
            logger.info(f"💾 Model saved: {name} -> {model_path}")

        # Save scaler
        scaler_path = model_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"💾 Scaler saved: {scaler_path}")

        return model_paths

    @staticmethod
    def load_models(model_dir: str) -> Dict[str, Any]:
        """Load trained models from disk."""
        model_dir = Path(model_dir)
        models = {}

        for model_file in model_dir.glob("*.joblib"):
            if model_file.name != "scaler.joblib":
                name = model_file.stem.replace('_', ' ').title()
                models[name] = joblib.load(model_file)
                logger.info(f"📂 Loaded model: {name}")

        return models
