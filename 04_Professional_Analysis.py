"""
Professional Coffee Cognition Analysis
======================================
A comprehensive data science pipeline demonstrating professional-grade
data processing, feature engineering, model selection, and evaluation.

Author: Data Science Professional
Date: December 2025
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, PowerTransformer

# Sklearn - Model Selection
from sklearn.model_selection import (
    train_test_split, KFold, RepeatedKFold, cross_val_score,
    cross_val_predict, GridSearchCV, RandomizedSearchCV, learning_curve
)

# Sklearn - Feature Selection
from sklearn.feature_selection import (
    RFE, SelectKBest, mutual_info_regression, f_regression
)
from sklearn.inspection import permutation_importance

# Sklearn - Linear Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, BayesianRidge
)

# Sklearn - Ensemble Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor,
    StackingRegressor, VotingRegressor
)

# Sklearn - Other Models
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Sklearn - Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Sklearn - Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced Models (install with: pip install xgboost lightgbm)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed. Run: pip install lightgbm")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("PROFESSIONAL COFFEE COGNITION ANALYSIS")
print("=" * 60)


# ============================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data(filepath):
    """
    Load data and perform comprehensive initial exploration.
    
    Parameters:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print("\n" + "="*60)
    print("SECTION 2: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns:\n{list(df.columns)}")
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n📈 Statistical Summary:")
    print(df.describe().round(3))
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    return df


def visualize_distributions(df, numeric_cols):
    """Create distribution plots for numeric columns."""
    print("\n📊 Creating Distribution Plots...")
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].set_xlabel('')
        
        # Add skewness annotation
        skew = df[col].skew()
        axes[idx].annotate(f'Skew: {skew:.2f}', xy=(0.7, 0.9), 
                          xycoords='axes fraction', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def detect_and_handle_outliers(df, numeric_cols, method='iqr', threshold=1.5):
    """
    Detect and handle outliers using multiple methods.
    
    Parameters:
        df: DataFrame
        numeric_cols: List of numeric column names
        method: 'iqr', 'zscore', or 'isolation_forest'
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    print(f"\n🔎 Outlier Detection (Method: {method.upper()})")
    
    df_clean = df.copy()
    outlier_report = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            
            # Winsorize outliers
            df_clean[col] = df[col].clip(lower=lower, upper=upper)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = (z_scores > 3).sum()
            
            # Replace outliers with median
            median = df[col].median()
            df_clean.loc[z_scores > 3, col] = median
        
        outlier_report[col] = outliers
    
    print("\n📋 Outlier Count per Feature:")
    for col, count in outlier_report.items():
        if count > 0:
            print(f"  • {col}: {count} outliers ({count/len(df)*100:.1f}%)")
    
    total_outliers = sum(outlier_report.values())
    if total_outliers == 0:
        print("  ✅ No significant outliers detected!")
    
    return df_clean


# ============================================================================
# SECTION 3: ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_advanced_features(df):
    """
    Create advanced engineered features for better model performance.
    
    Parameters:
        df: DataFrame
        
    Returns:
        DataFrame with new features
    """
    print("\n" + "="*60)
    print("SECTION 3: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    df_fe = df.copy()
    
    # 1. Polynomial Features for key numeric variables
    print("\n🔧 Creating Polynomial Features...")
    df_fe['Caffeine_sq'] = df_fe['Caffeine_mg'] ** 2
    df_fe['Age_sq'] = df_fe['Age'] ** 2
    
    # 2. Interaction Features
    print("🔧 Creating Interaction Features...")
    df_fe['Caffeine_x_Sleep'] = df_fe['Caffeine_mg'] * df_fe['Sleep_Hours']
    df_fe['Caffeine_x_SleepQuality'] = df_fe['Caffeine_mg'] * df_fe['Sleep_Quality_Score']
    df_fe['Caffeine_x_Stress'] = df_fe['Caffeine_mg'] * df_fe['Stress_Level']
    df_fe['Sleep_x_Stress'] = df_fe['Sleep_Hours'] * df_fe['Stress_Level']
    df_fe['Activity_x_Sleep'] = df_fe['Physical_Activity_Level'] * df_fe['Sleep_Hours']
    
    # 3. Ratio Features
    print("🔧 Creating Ratio Features...")
    df_fe['Sleep_Efficiency'] = df_fe['Sleep_Quality_Score'] / (df_fe['Sleep_Hours'] + 0.1)
    df_fe['Stress_per_Activity'] = df_fe['Stress_Level'] / (df_fe['Physical_Activity_Level'] + 0.1)
    
    # 4. Binned Features
    print("🔧 Creating Binned Features...")
    df_fe['Age_Group'] = pd.cut(df_fe['Age'], 
                                 bins=[0, 25, 35, 45, 55, 100], 
                                 labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    
    df_fe['Caffeine_Level'] = pd.cut(df_fe['Caffeine_mg'], 
                                      bins=[0, 100, 200, 300, 500],
                                      labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # 5. Log Transform for skewed features
    print("🔧 Applying Log Transforms...")
    for col in ['Caffeine_mg', 'Stroop_Reaction_Time_ms', 'PVT_Reaction_Time_ms']:
        if col in df_fe.columns:
            df_fe[f'{col}_log'] = np.log1p(df_fe[col])
    
    print(f"\n✅ Total Features After Engineering: {df_fe.shape[1]}")
    print(f"📋 New Features Created: {df_fe.shape[1] - df.shape[1]}")
    
    return df_fe


def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
        df: DataFrame
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with encoded features
    """
    print("\n🔧 Encoding Categorical Features...")
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"✅ Encoded {len(categorical_cols)} categorical features")
    print(f"📊 New Shape: {df_encoded.shape}")
    
    return df_encoded


def scale_features(X_train, X_test, numeric_cols, method='robust'):
    """
    Scale numeric features using specified method.
    
    Parameters:
        X_train, X_test: Train and test feature matrices
        numeric_cols: List of numeric column names
        method: 'standard', 'robust', or 'power'
        
    Returns:
        Scaled X_train, X_test
    """
    print(f"\n🔧 Scaling Features (Method: {method})...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = StandardScaler()
    
    # Only scale numeric columns that exist in the data
    cols_to_scale = [c for c in numeric_cols if c in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    print(f"✅ Scaled {len(cols_to_scale)} numeric features")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# SECTION 4: MODEL TRAINING AND EVALUATION
# ============================================================================

def get_models():
    """
    Return a dictionary of regression models to evaluate.
    
    Returns:
        dict: {model_name: model_instance}
    """
    models = {
        # Linear Models
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
        'HuberRegressor': HuberRegressor(max_iter=1000),
        'BayesianRidge': BayesianRidge(),
        
        # Tree-based Models
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=RANDOM_STATE, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10,
                                          random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                       learning_rate=0.1,
                                                       random_state=RANDOM_STATE),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE),
        
        # Other Models
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        )
    
    return models


def evaluate_models_cv(models, X, y, cv=5, scoring='neg_root_mean_squared_error'):
    """
    Evaluate multiple models using cross-validation.
    
    Parameters:
        models: Dictionary of models
        X: Feature matrix
        y: Target variable
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        DataFrame with model performance
    """
    print("\n" + "="*60)
    print("SECTION 4: MODEL EVALUATION (Cross-Validation)")
    print("="*60)
    
    results = []
    
    kf = RepeatedKFold(n_splits=cv, n_repeats=3, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\n🔄 Evaluating: {name}...", end=" ")
        
        try:
            # Cross-validation scores
            cv_rmse = -cross_val_score(model, X, y, cv=kf, 
                                        scoring='neg_root_mean_squared_error', 
                                        n_jobs=-1)
            cv_mae = -cross_val_score(model, X, y, cv=kf,
                                       scoring='neg_mean_absolute_error',
                                       n_jobs=-1)
            cv_r2 = cross_val_score(model, X, y, cv=kf,
                                    scoring='r2', n_jobs=-1)
            
            results.append({
                'Model': name,
                'RMSE_Mean': cv_rmse.mean(),
                'RMSE_Std': cv_rmse.std(),
                'MAE_Mean': cv_mae.mean(),
                'MAE_Std': cv_mae.std(),
                'R2_Mean': cv_r2.mean(),
                'R2_Std': cv_r2.std()
            })
            
            print(f"✅ R² = {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2_Mean', ascending=False)
    
    return results_df


def hyperparameter_tuning(model, param_grid, X, y, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters:
        model: Model instance
        param_grid: Parameter grid dictionary
        X: Feature matrix
        y: Target variable
        cv: Number of CV folds
        
    Returns:
        Best model and best parameters
    """
    print("\n🔧 Performing Hyperparameter Tuning...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\n✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Best Score (RMSE): {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def feature_importance_analysis(model, X, y, feature_names, top_n=15):
    """
    Analyze and visualize feature importance.
    
    Parameters:
        model: Trained model
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_type = "Feature Importance"
    else:
        # Use permutation importance for other models
        print("🔄 Calculating Permutation Importance...")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, 
                                                  random_state=RANDOM_STATE, n_jobs=-1)
        importance = perm_importance.importances_mean
        importance_type = "Permutation Importance"
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\n📊 Top {top_n} Features ({importance_type}):")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['Importance'].values)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - {importance_type}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return importance_df


# ============================================================================
# SECTION 5: ENSEMBLE METHODS
# ============================================================================

def create_stacking_ensemble(X, y):
    """
    Create a stacking ensemble model.
    
    Parameters:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Fitted stacking model
    """
    print("\n" + "="*60)
    print("SECTION 5: STACKING ENSEMBLE")
    print("="*60)
    
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, 
                                      random_state=RANDOM_STATE, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                          learning_rate=0.1, random_state=RANDOM_STATE)),
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        base_estimators.append(
            ('xgb', XGBRegressor(n_estimators=100, max_depth=5, 
                                 learning_rate=0.1, random_state=RANDOM_STATE,
                                 n_jobs=-1, verbosity=0))
        )
    
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    print("🔄 Training Stacking Ensemble...")
    stacking.fit(X, y)
    
    # Evaluate
    cv_scores = cross_val_score(stacking, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"\n✅ Stacking Ensemble R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return stacking


def create_voting_ensemble(X, y):
    """
    Create a voting ensemble model.
    
    Parameters:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Fitted voting model
    """
    print("\n🔄 Training Voting Ensemble...")
    
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10,
                                      random_state=RANDOM_STATE, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                          learning_rate=0.1, random_state=RANDOM_STATE)),
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]
    
    voting = VotingRegressor(estimators=estimators, n_jobs=-1)
    voting.fit(X, y)
    
    # Evaluate
    cv_scores = cross_val_score(voting, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"✅ Voting Ensemble R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return voting


# ============================================================================
# SECTION 6: VISUALIZATION AND REPORTING
# ============================================================================

def plot_model_comparison(results_df):
    """
    Create visualization comparing model performance.
    
    Parameters:
        results_df: DataFrame with model results
    """
    print("\n📊 Creating Model Comparison Plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² Score Comparison
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))
    bars = ax1.barh(results_df['Model'], results_df['R2_Mean'], 
                    xerr=results_df['R2_Std'], color=colors, capsize=3)
    ax1.set_xlabel('R² Score')
    ax1.set_title('Model Comparison (R²)', fontweight='bold')
    ax1.axvline(x=results_df['R2_Mean'].max(), color='red', linestyle='--', alpha=0.5)
    
    # RMSE Comparison
    ax2 = axes[1]
    ax2.barh(results_df['Model'], results_df['RMSE_Mean'],
             xerr=results_df['RMSE_Std'], color=colors, capsize=3)
    ax2.set_xlabel('RMSE')
    ax2.set_title('Model Comparison (RMSE)', fontweight='bold')
    
    # MAE Comparison
    ax3 = axes[2]
    ax3.barh(results_df['Model'], results_df['MAE_Mean'],
             xerr=results_df['MAE_Std'], color=colors, capsize=3)
    ax3.set_xlabel('MAE')
    ax3.set_title('Model Comparison (MAE)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_residual_analysis(model, X, y, model_name='Model'):
    """
    Create residual analysis plots.
    
    Parameters:
        model: Trained model
        X: Feature matrix
        y: True target values
        model_name: Name of the model
    """
    print(f"\n📊 Creating Residual Analysis for {model_name}...")
    
    y_pred = cross_val_predict(model, X, y, cv=5)
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    
    # Residual Distribution
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    
    # Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # Actual vs Predicted
    ax4 = axes[1, 1]
    ax4.scatter(y, y_pred, alpha=0.5, edgecolors='none')
    ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Predicted Values')
    ax4.set_title('Actual vs Predicted')
    
    plt.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_residuals.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_learning_curve(model, X, y, model_name='Model'):
    """
    Plot learning curve for a model.
    
    Parameters:
        model: Model instance
        X: Feature matrix
        y: Target variable
        model_name: Name of the model
    """
    print(f"\n📊 Creating Learning Curve for {model_name}...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_root_mean_squared_error'
    )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = -test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Score')
    
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_learning_curve.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def generate_summary_report(results_df, best_model_name, best_score):
    """
    Generate a summary report of the analysis.
    
    Parameters:
        results_df: DataFrame with model results
        best_model_name: Name of the best performing model
        best_score: Best R² score
    """
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    
    print(f"""
📊 ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 BEST MODEL: {best_model_name}
   R² Score: {best_score:.4f}

📈 MODEL RANKINGS (by R² Score):
""")
    
    for idx, row in results_df.head(5).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"   {rank}. {row['Model']}: R² = {row['R2_Mean']:.4f} (±{row['R2_Std']:.4f})")
    
    print(f"""
💡 KEY INSIGHTS:
   • Total models evaluated: {len(results_df)}
   • Best performing model: {best_model_name}
   • Model improvement over baseline: {(best_score - results_df['R2_Mean'].min())*100:.1f}%

📁 OUTPUT FILES GENERATED:
   • distributions.png - Feature distributions
   • model_comparison.png - Model comparison charts
   • feature_importance.png - Feature importance analysis
   • *_residuals.png - Residual analysis plots
   • *_learning_curve.png - Learning curves

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the complete analysis pipeline.
    """
    # Step 1: Load Data
    df = load_and_explore_data("coffee_cognition_realistic_700.csv")
    
    # Define columns
    target_cols = ['Stroop_Reaction_Time_ms', 'PVT_Reaction_Time_ms', 
                   'N_Back_Accuracy', 'Focus_Level']
    categorical_cols = ['Gender', 'Brewing_Method', 'Time_of_Day']
    id_cols = ['Participant_ID']
    
    numeric_cols = [col for col in df.columns 
                    if col not in target_cols + categorical_cols + id_cols]
    
    # Step 2: Visualize distributions
    visualize_distributions(df, numeric_cols + target_cols[:2])
    
    # Step 3: Handle outliers
    df_clean = detect_and_handle_outliers(df, numeric_cols, method='iqr')
    
    # Step 4: Feature Engineering
    df_fe = create_advanced_features(df_clean)
    
    # Step 5: Encode categorical features
    categorical_cols_extended = categorical_cols + ['Age_Group', 'Caffeine_Level']
    df_encoded = encode_categorical_features(df_fe, categorical_cols_extended)
    
    # Step 6: Prepare features and target
    # Let's focus on predicting Focus_Level as our primary target
    target = 'Focus_Level'
    
    feature_cols = [col for col in df_encoded.columns 
                    if col not in target_cols + id_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target]
    
    print(f"\n📊 Feature Matrix Shape: {X.shape}")
    print(f"📊 Target Variable: {target}")
    
    # Step 7: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Step 8: Scale features
    numeric_feature_cols = [col for col in numeric_cols if col in X.columns]
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, numeric_feature_cols, method='robust'
    )
    
    # Step 9: Get and evaluate models
    models = get_models()
    results_df = evaluate_models_cv(models, X_train_scaled, y_train, cv=5)
    
    # Step 10: Display results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Step 11: Visualize model comparison
    plot_model_comparison(results_df)
    
    # Step 12: Train best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Step 13: Hyperparameter tuning for best model (if RandomForest or GradientBoosting)
    if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
        if best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        elif best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        elif best_model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        elif best_model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        else:
            param_grid = None
        
        if param_grid:
            best_model, best_params = hyperparameter_tuning(
                models[best_model_name], param_grid, X_train_scaled, y_train
            )
    
    # Step 14: Feature importance analysis
    best_model.fit(X_train_scaled, y_train)
    feature_importance_analysis(best_model, X_train_scaled, y_train, 
                                X_train_scaled.columns.tolist())
    
    # Step 15: Residual analysis
    plot_residual_analysis(best_model, X_train_scaled, y_train, best_model_name)
    
    # Step 16: Learning curve
    plot_learning_curve(best_model, X_train_scaled, y_train, best_model_name)
    
    # Step 17: Ensemble methods
    stacking_model = create_stacking_ensemble(X_train_scaled, y_train)
    voting_model = create_voting_ensemble(X_train_scaled, y_train)
    
    # Step 18: Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\n🎯 Test Set Results for {best_model_name}:")
    print(f"   • RMSE: {test_rmse:.4f}")
    print(f"   • MAE: {test_mae:.4f}")
    print(f"   • R²: {test_r2:.4f}")
    
    # Step 19: Generate summary report
    generate_summary_report(results_df, best_model_name, results_df.iloc[0]['R2_Mean'])
    
    print("\n✅ Analysis Complete!")
    
    return results_df, best_model


# Run the main analysis
if __name__ == "__main__":
    results, best_model = main()
