import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


print("ACCIDENT PREDICTION MODEL - QUARTERLY FORECASTING (REVISED)")


# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================


print("STEP 1: DATA LOADING AND PREPROCESSING")


# Load data
df = pd.read_csv('full_data_v2.csv')
print(f" Loaded {len(df)} total records")

# REMOVE mile344 segment
df = df[df['28milename'] != 'mile344'].copy()
print(f" Removed mile344 segment")
print(f" Remaining records: {len(df)}")

# Add Quarter column
def get_quarter(month):
    """Convert month to quarter"""
    if pd.isna(month):
        return None
    month = int(month)
    if month in [1, 2, 3]:
        return 'Q1'
    elif month in [4, 5, 6]:
        return 'Q2'
    elif month in [7, 8, 9]:
        return 'Q3'
    else:
        return 'Q4'

df['Quarter'] = df['CRASH_MONTH'].apply(get_quarter)
print(" Added Quarter column")

# Filter for valid data
df_valid = df[
    df['28milename'].notna() & 
    df['CRASH_YEAR'].notna() & 
    df['Quarter'].notna()
].copy()

print(f" Valid records with segment assignment: {len(df_valid)}")
print(f"\nYears in data: {sorted(df_valid['CRASH_YEAR'].unique())}")
print(f"Segments: {sorted(df_valid['28milename'].unique())}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================


print("STEP 2: FEATURE ENGINEERING")


# Create base aggregation
agg_dict = {
    'CRN': 'count',  # Total accidents
    '14milelat': 'first',
    '14milelon': 'first',
    'FATAL_COUNT': 'sum',
    'TOT_INJ_COUNT': 'sum',
}

# Aggregate by segment, year, and quarter
quarterly_data = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter']).agg(agg_dict).reset_index()
quarterly_data.rename(columns={'CRN': 'total_accidents'}, inplace=True)

print(f" Aggregated to {len(quarterly_data)} segment-quarter records")

# Add severe accidents count (MAX_SEVERITY_LEVEL = 1 or 2)
severe_accidents = df_valid[df_valid['MAX_SEVERITY_LEVEL'].isin([1, 2])].groupby(
    ['28milename', 'CRASH_YEAR', 'Quarter']
).size().reset_index(name='severe_accidents')

quarterly_data = quarterly_data.merge(
    severe_accidents, 
    on=['28milename', 'CRASH_YEAR', 'Quarter'], 
    how='left'
)
quarterly_data['severe_accidents'] = quarterly_data['severe_accidents'].fillna(0)

print(" Added severe accidents count")

# Add road condition features
road_conditions = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter'])['ROAD_CONDITION'].agg([
    ('poor_road_pct', lambda x: (x.isin([2, 3, 4])).sum() / len(x) if len(x) > 0 else 0)
]).reset_index()

quarterly_data = quarterly_data.merge(road_conditions, on=['28milename', 'CRASH_YEAR', 'Quarter'], how='left')
quarterly_data['poor_road_pct'] = quarterly_data['poor_road_pct'].fillna(0)
print(" Added road condition features")

# Add collision type diversity
collision_diversity = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter'])['COLLISION_TYPE'].nunique().reset_index()
collision_diversity.columns = ['28milename', 'CRASH_YEAR', 'Quarter', 'collision_type_diversity']

quarterly_data = quarterly_data.merge(collision_diversity, on=['28milename', 'CRASH_YEAR', 'Quarter'], how='left')
quarterly_data['collision_type_diversity'] = quarterly_data['collision_type_diversity'].fillna(1)
print(" Added collision type diversity")

# Convert quarter to numeric for modeling
quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
quarterly_data['quarter_num'] = quarterly_data['Quarter'].map(quarter_map)

# Add segment identifier (numeric)
segment_map = {seg: idx for idx, seg in enumerate(sorted(quarterly_data['28milename'].unique()))}
quarterly_data['segment_id'] = quarterly_data['28milename'].map(segment_map)

# Add winter indicator (Q1 and Q4 are winter)
quarterly_data['is_winter'] = quarterly_data['quarter_num'].isin([1, 4]).astype(int)

print("\n Feature Engineering Summary:")
print(f"  - Total accidents per segment-quarter")
print(f"  - Severe accidents (MAX_SEVERITY_LEVEL = 1 or 2)")
print(f"  - Fatalities and injuries")
print(f"  - Road condition percentage")
print(f"  - Collision type diversity")
print(f"  - Temporal features (year, quarter, winter indicator)")
print(f"  - Spatial features (segment ID, lat/lon)")

# ============================================================================
# STEP 3: CREATE TIME-BASED FEATURES (LAG FEATURES)
# ============================================================================


print("STEP 3: CREATING TIME-BASED LAG FEATURES")


# Sort by segment and time
quarterly_data = quarterly_data.sort_values(['28milename', 'CRASH_YEAR', 'quarter_num'])

# Create lag features (previous quarters)
for segment in quarterly_data['28milename'].unique():
    mask = quarterly_data['28milename'] == segment
    
    # Lag 1 (previous quarter)
    quarterly_data.loc[mask, 'accidents_lag1'] = quarterly_data.loc[mask, 'total_accidents'].shift(1)
    
    # Lag 4 (same quarter last year)
    quarterly_data.loc[mask, 'accidents_lag4'] = quarterly_data.loc[mask, 'total_accidents'].shift(4)
    
    # Rolling mean (last 4 quarters)
    quarterly_data.loc[mask, 'accidents_rolling_mean'] = quarterly_data.loc[mask, 'total_accidents'].shift(1).rolling(window=4, min_periods=1).mean()
    
    # Rolling std (variability)
    quarterly_data.loc[mask, 'accidents_rolling_std'] = quarterly_data.loc[mask, 'total_accidents'].shift(1).rolling(window=4, min_periods=2).std()

# Fill missing values with segment-specific medians
for col in ['accidents_lag1', 'accidents_lag4', 'accidents_rolling_mean', 'accidents_rolling_std']:
    segment_medians = quarterly_data.groupby('segment_id')[col].median()
    for segment_id in quarterly_data['segment_id'].unique():
        mask = (quarterly_data['segment_id'] == segment_id) & (quarterly_data[col].isna())
        quarterly_data.loc[mask, col] = segment_medians.get(segment_id, quarterly_data[col].median())

quarterly_data['accidents_rolling_std'] = quarterly_data['accidents_rolling_std'].fillna(0)

print(" Created lag features:")
print("  - Previous quarter accidents")
print("  - Same quarter last year")
print("  - 4-quarter rolling average")
print("  - 4-quarter rolling std (variability)")

# Add segment-level baseline features
segment_baselines = quarterly_data.groupby('segment_id').agg({
    'total_accidents': ['mean', 'std']
}).reset_index()
segment_baselines.columns = ['segment_id', 'segment_mean_accidents', 'segment_std_accidents']
quarterly_data = quarterly_data.merge(segment_baselines, on='segment_id', how='left')
quarterly_data['segment_std_accidents'] = quarterly_data['segment_std_accidents'].fillna(0)

print(" Added segment baseline features")

# Display sample
print("\nSample of processed data:")
display_cols = ['28milename', 'CRASH_YEAR', 'Quarter', 'total_accidents', 'severe_accidents', 
                'accidents_lag1', 'accidents_rolling_mean']
print(quarterly_data[display_cols].head(15).to_string(index=False))

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================


print("STEP 4: TRAIN/TEST SPLIT")


# Split data
train_data = quarterly_data[quarterly_data['CRASH_YEAR'].isin([2022, 2023])].copy()
test_data = quarterly_data[quarterly_data['CRASH_YEAR'] == 2024].copy()

print(f" Training data: {len(train_data)} records (2022-2023)")
print(f" Test data: {len(test_data)} records (2024)")
print(f"\nTraining data distribution by quarter:")
print(train_data.groupby('Quarter')['total_accidents'].agg(['count', 'mean', 'std']))

# Define features
feature_cols = [
    'quarter_num',
    'is_winter',
    'segment_id',
    'accidents_lag1',
    'accidents_lag4',
    'accidents_rolling_mean',
    'accidents_rolling_std',
    'segment_mean_accidents',
    'segment_std_accidents',
    'severe_accidents',
    'poor_road_pct',
    'collision_type_diversity'
]

X_train = train_data[feature_cols]
y_train = train_data['total_accidents']

X_test = test_data[feature_cols]
y_test = test_data['total_accidents']

print(f"\n Feature matrix shape: {X_train.shape}")
print(f" Features: {', '.join(feature_cols)}")

# Check for any remaining NaN values
if X_train.isna().any().any() or X_test.isna().any().any():
    print("\n⚠ Found NaN values, filling with median...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    print(" NaN values filled")

# ============================================================================
# STEP 5: BASE MODEL - RANDOM FOREST
# ============================================================================


print("STEP 5: BASE MODEL - RANDOM FOREST")


# Train base model with conservative parameters
base_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

base_model.fit(X_train, y_train)
print(" Base model trained")

# Cross-validation on training set
cv_scores = cross_val_score(base_model, X_train, y_train, cv=3, 
                            scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores.mean())
print(f" Cross-validation RMSE: {cv_rmse:.2f}")

# Predictions
y_pred_base = base_model.predict(X_test)
y_pred_base = np.maximum(y_pred_base, 0)  # No negative predictions

# Metrics
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print("\n BASE MODEL PERFORMANCE:")
print(f"  RMSE: {rmse_base:.2f}")
print(f"  MAE: {mae_base:.2f}")
print(f"  R²: {r2_base:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': base_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n FEATURE IMPORTANCE (Top 5):")
for idx, row in feature_importance.head().iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# STEP 6: IMPROVED MODEL - GRADIENT BOOSTING (SIMPLIFIED)
# ============================================================================


print("STEP 6: IMPROVED MODEL - GRADIENT BOOSTING (OPTIMIZED)")


# More conservative parameter grid to avoid overfitting
param_candidates = [
    # Conservative
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_samples_split': 10, 'subsample': 0.8},
    # Moderate
    {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05, 'min_samples_split': 8, 'subsample': 0.8},
    # Slightly aggressive
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.03, 'min_samples_split': 6, 'subsample': 0.8},
]

best_model = None
best_score = float('inf')
best_params = None

print(" Testing parameter combinations with cross-validation...")

for i, params in enumerate(param_candidates, 1):
    model = GradientBoostingRegressor(random_state=42, **params)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                                scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"\n  Candidate {i}: CV RMSE = {cv_rmse:.2f}")
    print(f"    Parameters: {params}")
    
    if cv_rmse < best_score:
        best_score = cv_rmse
        best_model = model
        best_params = params

print(f"\n Best model selected (CV RMSE: {best_score:.2f})")
print(f"  Best parameters: {best_params}")

# Train final model on full training set
final_model = GradientBoostingRegressor(random_state=42, **best_params)
final_model.fit(X_train, y_train)

# Predictions
y_pred_final = final_model.predict(X_test)
y_pred_final = np.maximum(y_pred_final, 0)

# Metrics
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
mae_final = mean_absolute_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

print("\n IMPROVED MODEL PERFORMANCE:")
print(f"  RMSE: {rmse_final:.2f}")
print(f"  MAE: {mae_final:.2f}")
print(f"  R²: {r2_final:.3f}")

# Compare models
print("\n MODEL COMPARISON:")
if rmse_final < rmse_base:
    improvement = rmse_base - rmse_final
    pct_improvement = (improvement / rmse_base) * 100
    print(f"   Improved model is BETTER by {improvement:.2f} RMSE ({pct_improvement:.1f}% improvement)")
else:
    print(f"   Base model performs slightly better (RMSE: {rmse_base:.2f} vs {rmse_final:.2f})")
    print(f"   Using base model for final predictions")
    final_model = base_model
    y_pred_final = y_pred_base
    rmse_final = rmse_base
    mae_final = mae_base
    r2_final = r2_base

# ============================================================================
# STEP 7: ALTERNATIVE MODEL - RIDGE REGRESSION (BENCHMARK)
# ============================================================================


print("STEP 7: RIDGE REGRESSION BENCHMARK")


ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)
y_pred_ridge = np.maximum(y_pred_ridge, 0)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"RIDGE REGRESSION PERFORMANCE:")
print(f"  RMSE: {rmse_ridge:.2f}")
print(f"  MAE: {mae_ridge:.2f}")
print(f"  R²: {r2_ridge:.3f}")

# ============================================================================
# STEP 8: RESULTS VISUALIZATION
# ============================================================================


print("STEP 8: CREATING VISUALIZATIONS")


# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Actual vs Predicted (Base Model)
axes[0, 0].scatter(y_test, y_pred_base, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Accidents', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Accidents', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Base Model (Random Forest)\nRMSE={rmse_base:.2f}, MAE={mae_base:.2f}, R²={r2_base:.3f}', 
                     fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Best Model)
best_model_name = "Gradient Boosting" if rmse_final < rmse_base else "Random Forest"
axes[0, 1].scatter(y_test, y_pred_final, alpha=0.6, s=100, color='green', edgecolors='black', linewidth=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Accidents', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Predicted Accidents', fontsize=12, fontweight='bold')
axes[0, 1].set_title(f'Best Model ({best_model_name})\nRMSE={rmse_final:.2f}, MAE={mae_final:.2f}, R²={r2_final:.3f}', 
                     fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Best Model)
if hasattr(final_model, 'feature_importances_'):
    feature_importance_final = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1, 0].barh(feature_importance_final['feature'], feature_importance_final['importance'], color='steelblue')
    axes[1, 0].set_xlabel('Importance', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Feature Importance (Best Model)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
else:
    axes[1, 0].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                   ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('Feature Importance', fontsize=14, fontweight='bold')

# 4. Model Comparison
models = ['Random Forest\n(Base)', 'Gradient Boosting', 'Ridge Regression']
rmses = [rmse_base, rmse_final if 'Gradient' in best_model_name else rmse_base, rmse_ridge]
maes = [mae_base, mae_final if 'Gradient' in best_model_name else mae_base, mae_ridge]

x = np.arange(len(models))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, rmses, width, label='RMSE', alpha=0.8)
bars2 = axes[1, 1].bar(x + width/2, maes, width, label='MAE', alpha=0.8)

axes[1, 1].set_ylabel('Error', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Model Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print(" Saved model_performance.png")

# ============================================================================
# STEP 9: DETAILED PREDICTIONS TABLE
# ============================================================================


print("STEP 9: PREDICTION RESULTS")


# Create detailed results
results_df = test_data[['28milename', 'CRASH_YEAR', 'Quarter', 'total_accidents']].copy()
results_df['predicted'] = y_pred_final
results_df['error'] = results_df['total_accidents'] - results_df['predicted']
results_df['abs_error'] = results_df['error'].abs()
results_df['pct_error'] = (results_df['error'] / results_df['total_accidents'] * 100).replace([np.inf, -np.inf], 0)

# Save results
results_df.to_csv('quarterly_predictions_2024.csv', index=False)
print(" Saved quarterly_predictions_2024.csv")

# Show summary by segment
print("\nPrediction Summary by Segment:")
segment_summary = results_df.groupby('28milename').agg({
    'total_accidents': 'sum',
    'predicted': 'sum',
    'abs_error': 'mean'
}).round(1)
segment_summary.columns = ['Actual', 'Predicted', 'Avg Error']
print(segment_summary)

# Show summary by quarter
print("\nPrediction Summary by Quarter:")
quarter_summary = results_df.groupby('Quarter').agg({
    'total_accidents': 'sum',
    'predicted': 'sum'
}).round(1)
quarter_summary['error'] = quarter_summary['total_accidents'] - quarter_summary['predicted']
quarter_summary['error_pct'] = (quarter_summary['error'] / quarter_summary['total_accidents'] * 100).round(1)
print(quarter_summary)

# ============================================================================
# STEP 10: SAVE TRAINED MODEL
# ============================================================================


print("STEP 10: SAVING MODEL")


import pickle

# Save the best model
with open('accident_prediction_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print(" Saved accident_prediction_model.pkl")

# Save feature columns
with open('model_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(" Saved model_features.pkl")

# Save segment mapping
with open('segment_mapping.pkl', 'wb') as f:
    pickle.dump(segment_map, f)
print(" Saved segment_mapping.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================


print("FINAL SUMMARY")


print(f"\n BEST MODEL: {best_model_name}")
print(f"  RMSE: {rmse_final:.2f}")
print(f"  MAE: {mae_final:.2f}")
print(f"  R²: {r2_final:.3f}")

print("\nFILES CREATED:")
print("  1. model_performance.png - Visualization of results")
print("  2. quarterly_predictions_2024.csv - Detailed predictions")
print("  3. accident_prediction_model.pkl - Trained model")
print("  4. model_features.pkl - Feature list")
print("  5. segment_mapping.pkl - Segment ID mapping")

print(f"  - Removed mile344 from analysis")
print(f"  - Analyzed {len(quarterly_data['28milename'].unique())} segments")
print(f"  - Trained on {len(train_data)} quarters (2022-2023)")
print(f"  - Tested on {len(test_data)} quarters (2024)")
print(f"  - Average prediction error: {mae_final:.2f} accidents per quarter")


print("ANALYSIS COMPLETE!")
