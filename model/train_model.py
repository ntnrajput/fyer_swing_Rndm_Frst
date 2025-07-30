# model/train_model.py

import sys

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import warnings
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def validate_features(df):
    """Validate and report on available features"""
    logger.info("Validating available features...")
    
    # Focus on core feature categories only
    core_features = {
        'Technical Indicators': ['rsi', 'atr', 'bb_position'],
        'Price Movement': ['daily_return', 'range_pct'],
        'Volume': ['volume_ratio'],
        'Trend': ['trend_strength'],
        'Support/Resistance': ['norm_dist_to_support', 'norm_dist_to_resistance']
    }
    
    available_features = {}
    for category, features in core_features.items():
        available = [f for f in features if f in df.columns]
        if available:
            available_features[category] = available
    
    # Report available features
    logger.info("Available Core Features:")
    for category, features in available_features.items():
        logger.info(f"  {category}: {features}")
    
    total_available = sum(len(features) for features in available_features.values())
    logger.info(f"Total Core Features Available: {total_available}")
    
    return available_features

def create_basic_features(df):
    """Create only essential features to reduce overfitting"""
    logger.info("Creating basic features...")
    
    # Only add features that don't already exist
    existing_features = set(df.columns)
    
    # Essential price features
    if 'daily_return' not in existing_features:
        df['daily_return'] = df['close'].pct_change()
    
    if 'range_pct' not in existing_features:
        df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Simple volume feature
    if 'volume' in df.columns and 'volume_ratio' not in existing_features:
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    
    # Basic trend feature
    if 'trend_strength' not in existing_features:
        df['price_ma_10'] = df['close'].rolling(10).mean()
        df['price_ma_20'] = df['close'].rolling(20).mean()
        df['trend_strength'] = (df['price_ma_10'] - df['price_ma_20']) / df['close']
    
    # Simple RSI-based features (if RSI exists)
    if 'rsi' in existing_features:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    return df

def create_simple_model(X_train, y_train, class_weight_dict=None):
    """Create a simple, less complex model to reduce overfitting"""
    logger.info("Creating simple model...")
    
    # Simple Random Forest with conservative parameters
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100-300
        max_depth=5,      # Limited depth
        min_samples_split=10,  # Increased to prevent overfitting
        min_samples_leaf=5,    # Increased to prevent overfitting
        max_features='sqrt',   # Reduced feature subset
        class_weight=class_weight_dict,
        random_state=42
    )
    
    # Simple Logistic Regression as backup
    lr_model = LogisticRegression(
        C=1.0,  # Not too regularized, not too complex
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42
    )
    
    # Use cross-validation to select best model
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    logger.info(f"RF Cross-validation AUC: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")
    logger.info(f"LR Cross-validation AUC: {lr_scores.mean():.3f} (+/- {lr_scores.std() * 2:.3f})")
    
    # Select best model
    if rf_scores.mean() > lr_scores.mean():
        logger.info("Selected Random Forest model")
        best_model = rf_model
    else:
        logger.info("Selected Logistic Regression model")
        best_model = lr_model
    
    best_model.fit(X_train, y_train)
    return best_model

def evaluate_model_performance(model, X_test, y_test, X_train, y_train):
    """Simplified model evaluation"""
    logger.info("Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Trading metrics
    true_positives = conf_matrix[1, 1]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    win_rate = precision * 100
    
    # Check for overfitting
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting_gap = train_score - test_score
    
    logger.info(f"\nMODEL PERFORMANCE METRICS:")
    logger.info(f"  Win Rate: {win_rate:.1f}%")
    logger.info(f"  AUC Score: {auc_score:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    logger.info(f"  Train Accuracy: {train_score:.3f}")
    logger.info(f"  Test Accuracy: {test_score:.3f}")
    logger.info(f"  Overfitting Gap: {overfitting_gap:.3f}")
    
    if overfitting_gap > 0.1:
        logger.warning("  WARNING: Potential overfitting detected!")
    else:
        logger.info("  Model shows good generalization")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    return {
        'auc_score': auc_score,
        'win_rate': win_rate,
        'precision': precision,
        'recall': recall,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'overfitting_gap': overfitting_gap,
        'classification_report': report
    }



def train_model(df_all, available_features=None, model_params=None):
    """
    Enhanced trading model with hyperparameter tuning and volume analysis
    
    Args:
        df_all: DataFrame with all features and target
        available_features: List of feature column names (optional, can be imported from config)
        model_params: Optional hyperparameter grid for tuning
    
    Returns:
        tuple: (best_model, scaler, available_features)
    """
    # Import FEATURE_COLUMNS if not provided
    if available_features is None:
        try:
            from config import FEATURE_COLUMNS
            available_features = FEATURE_COLUMNS
        except ImportError:
            raise ValueError("available_features must be provided or FEATURE_COLUMNS must be available in config")
    
    # Validate that all features exist in the dataframe
    missing_features = [col for col in available_features if col not in df_all.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    # Also check if target column exists
    if 'target_hit' not in df_all.columns:
        raise ValueError("Target column 'target_hit' not found in dataframe")
    try:
        # Data preparation - handle NaN values in both features and target
        print("📊 Initial data shape:", df_all.shape)
        
        # First, drop rows where target is NaN
        df_all = df_all.dropna(subset=['target_hit'])
        print("📊 After dropping NaN targets:", df_all.shape)
        
        # Then drop rows where any of the features are NaN
        df_all = df_all.dropna(subset=available_features)
        print("After dropping NaN features:", df_all.shape)
        
        X = df_all[available_features]
        y = df_all['target_hit']
        
        # Fill any remaining NaN values in features
        X = X.fillna(0)
        
        # Verify no NaN values remain
        print(f" NaN values in features: {X.isnull().sum().sum()}")
        print(f"NaN values in target: {y.isnull().sum()}")
        
        # Check class distribution
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            logger.error("Only one class found in target variable")
            raise ValueError("Only one class found in target variable")
            
        # Check if we have enough samples after cleaning
        if len(df_all) < 50:
            logger.warning(f"Very few samples remaining after cleaning: {len(df_all)}")
            
        # Additional check for target variable data types
        if y.dtype not in ['int64', 'int32', 'float64', 'float32', 'bool']:
            logger.warning(f"Target variable dtype is {y.dtype}, converting to numeric")
            y = pd.to_numeric(y, errors='coerce')
            # Drop any rows that couldn't be converted
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            print(f"After numeric conversion: {len(X)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ------------------------------------

        selector = SelectKBest(score_func=f_classif, k=15)  # You can change k
        X_selected = selector.fit_transform(X_scaled, y)

        # Get names of selected features
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]
        print(f" Selected Top {len(selected_features)} Features: {selected_features}")
        
        # Split data with stratification (replaces time-aware split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, stratify=y, test_size=0.2, random_state=42
        )

        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Enhanced hyperparameter tuning
        if model_params is None:
            model_params = {
                'n_estimators': [100],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1],
                'max_features': ['sqrt']
            }
        
        print("🔍 Performing enhanced hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, model_params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        print("\n Enhanced Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Separate volume features importance
        volume_features = feature_importance[feature_importance['feature'].str.contains('volume|vwap|obv|ad_line')]
        print(f"\n Volume Features Importance (Top 10):")
        print(volume_features.head(10))
        
        # Plot feature importance and save instead of showing
        try:
            plt.figure(figsize=(12, 10))                        
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('Enhanced Feature Importance with Volume Analysis')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print("📊 Feature importance plot saved as 'feature_importance_plot.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")
        
        # Calculate additional metrics for comparison with original
        train_pred = best_model.predict(X_train)
        train_accuracy = np.mean(train_pred == y_train)
        test_accuracy = np.mean(y_pred == y_test)
        overfitting_gap = train_accuracy - test_accuracy
        win_rate = test_accuracy * 100
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'win_rate': win_rate,
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Model metrics: {metrics}")
        
        # Create enhanced model pipeline
        model_pipeline = {
            'scaler': scaler,
            'selector': selector,
            'model': best_model,
            'feature_columns': selected_features,
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'feature_importance': feature_importance
        }
        
        # Save model and scaler
        joblib.dump(best_model, "enhanced_swing_model_with_volume.pkl")
        full_pipeline = Pipeline([
            ('scaler', scaler),
            ('selector', selector),
            ('model', best_model)
        ])

        # Bundle full model with metadata
        model_bundle = {
                'pipeline': full_pipeline,
                'feature_columns': selected_features,     # Features after selection
                'all_features': available_features,       # Features before selection (needed for backtest)
                'metrics': metrics,
                'best_params': grid_search.best_params_,
                'feature_importance': feature_importance
            }
        # Save single model bundle
        joblib.dump(model_bundle, MODEL_FILE)
        logger.info(f" Model pipeline saved to {MODEL_FILE}")
        




        print("Enhanced model with volume analysis saved!")
        
        # Trading recommendations
        logger.info("\nTRADING RECOMMENDATIONS:")
        if overfitting_gap < 0.05:
            logger.info("  Model shows good generalization")
        else:
            logger.warning("  Model may still be overfitting - use with caution")
        
        if win_rate >= 55:
            logger.info("  Model shows reasonable predictive power")
        else:
            logger.info("  Model needs improvement - consider more data or different features")
        
        logger.info(f"  Enhanced approach with ROC AUC: {metrics['roc_auc']:.4f} (win rate: {win_rate:.1f}%)")
        
        return model_bundle
        
    except Exception as e:
        logger.error(f"Failed to train enhanced model: {e}", exc_info=True)
        raise

