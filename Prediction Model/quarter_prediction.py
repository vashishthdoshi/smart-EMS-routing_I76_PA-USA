import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


print("PREDICTING ACCIDENTS FOR NEXT QUARTER")


# ============================================================================
# STEP 1: LOAD MODEL AND DATA
# ============================================================================


print("STEP 1: LOADING MODEL AND DATA")


# Load the trained model
with open('accident_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(" Loaded trained model")

# Load feature columns
with open('model_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
print(f" Loaded {len(feature_cols)} features")

# Load segment mapping
with open('segment_mapping.pkl', 'rb') as f:
    segment_map = pickle.load(f)
print(f" Loaded segment mapping ({len(segment_map)} segments)")

# Load the data
df = pd.read_csv('full_data_v2.csv')
print(f" Loaded {len(df)} records")

# Remove mile344
df = df[df['28milename'] != 'mile344'].copy()
print(" Removed mile344")

# ============================================================================
# STEP 2: PREPARE DATA AND DETERMINE NEXT QUARTER
# ============================================================================


print("STEP 2: DETERMINING NEXT QUARTER")


# Add Quarter column
def get_quarter(month):
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

# Filter valid data
df_valid = df[
    df['28milename'].notna() & 
    df['CRASH_YEAR'].notna() & 
    df['Quarter'].notna()
].copy()

# Find the most recent quarter in the data
latest_year = int(df_valid['CRASH_YEAR'].max())
latest_quarters = df_valid[df_valid['CRASH_YEAR'] == latest_year]['Quarter'].unique()

quarter_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
latest_quarter_num = max([quarter_order[q] for q in latest_quarters])
latest_quarter = [q for q in latest_quarters if quarter_order[q] == latest_quarter_num][0]

# Calculate next quarter
next_quarter_num = latest_quarter_num + 1
if next_quarter_num > 4:
    next_quarter_num = 1
    next_year = latest_year + 1
else:
    next_year = latest_year

next_quarter = f'Q{next_quarter_num}'

print(f" Latest data: {latest_quarter} {latest_year}")
print(f" Predicting for: {next_quarter} {next_year}")

# ============================================================================
# STEP 3: AGGREGATE HISTORICAL DATA
# ============================================================================


print("STEP 3: PREPARING HISTORICAL DATA")


# Create aggregations (same as training)
quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
df_valid['quarter_num'] = df_valid['Quarter'].map(quarter_map)

agg_dict = {
    'CRN': 'count',
    '14milelat': 'first',
    '14milelon': 'first',
    'FATAL_COUNT': 'sum',
    'TOT_INJ_COUNT': 'sum',
}

quarterly_data = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter']).agg(agg_dict).reset_index()
quarterly_data.rename(columns={'CRN': 'total_accidents'}, inplace=True)

# Add severe accidents
severe_accidents = df_valid[df_valid['MAX_SEVERITY_LEVEL'].isin([1, 2])].groupby(
    ['28milename', 'CRASH_YEAR', 'Quarter']
).size().reset_index(name='severe_accidents')

quarterly_data = quarterly_data.merge(
    severe_accidents, 
    on=['28milename', 'CRASH_YEAR', 'Quarter'], 
    how='left'
)
quarterly_data['severe_accidents'] = quarterly_data['severe_accidents'].fillna(0)

# Add road conditions
road_conditions = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter'])['ROAD_CONDITION'].agg([
    ('poor_road_pct', lambda x: (x.isin([2, 3, 4])).sum() / len(x) if len(x) > 0 else 0)
]).reset_index()

quarterly_data = quarterly_data.merge(road_conditions, on=['28milename', 'CRASH_YEAR', 'Quarter'], how='left')
quarterly_data['poor_road_pct'] = quarterly_data['poor_road_pct'].fillna(0)

# Add collision diversity
collision_diversity = df_valid.groupby(['28milename', 'CRASH_YEAR', 'Quarter'])['COLLISION_TYPE'].nunique().reset_index()
collision_diversity.columns = ['28milename', 'CRASH_YEAR', 'Quarter', 'collision_type_diversity']

quarterly_data = quarterly_data.merge(collision_diversity, on=['28milename', 'CRASH_YEAR', 'Quarter'], how='left')
quarterly_data['collision_type_diversity'] = quarterly_data['collision_type_diversity'].fillna(1)

quarterly_data['quarter_num'] = quarterly_data['Quarter'].map(quarter_map)
quarterly_data['segment_id'] = quarterly_data['28milename'].map(segment_map)
quarterly_data['is_winter'] = quarterly_data['quarter_num'].isin([1, 4]).astype(int)

print(" Aggregated historical data")

# ============================================================================
# STEP 4: CALCULATE LAG FEATURES FOR EACH SEGMENT
# ============================================================================


print("STEP 4: CALCULATING LAG FEATURES FOR NEXT QUARTER")


# Sort by segment and time
quarterly_data = quarterly_data.sort_values(['28milename', 'CRASH_YEAR', 'quarter_num'])

# Create lag features
for segment in quarterly_data['28milename'].unique():
    mask = quarterly_data['28milename'] == segment
    
    quarterly_data.loc[mask, 'accidents_lag1'] = quarterly_data.loc[mask, 'total_accidents'].shift(1)
    quarterly_data.loc[mask, 'accidents_lag4'] = quarterly_data.loc[mask, 'total_accidents'].shift(4)
    quarterly_data.loc[mask, 'accidents_rolling_mean'] = quarterly_data.loc[mask, 'total_accidents'].shift(1).rolling(window=4, min_periods=1).mean()
    quarterly_data.loc[mask, 'accidents_rolling_std'] = quarterly_data.loc[mask, 'total_accidents'].shift(1).rolling(window=4, min_periods=2).std()

# Fill missing values
for col in ['accidents_lag1', 'accidents_lag4', 'accidents_rolling_mean', 'accidents_rolling_std']:
    segment_medians = quarterly_data.groupby('segment_id')[col].median()
    for segment_id in quarterly_data['segment_id'].unique():
        mask = (quarterly_data['segment_id'] == segment_id) & (quarterly_data[col].isna())
        quarterly_data.loc[mask, col] = segment_medians.get(segment_id, quarterly_data[col].median())

quarterly_data['accidents_rolling_std'] = quarterly_data['accidents_rolling_std'].fillna(0)

# Add segment baselines
segment_baselines = quarterly_data.groupby('segment_id').agg({
    'total_accidents': ['mean', 'std']
}).reset_index()
segment_baselines.columns = ['segment_id', 'segment_mean_accidents', 'segment_std_accidents']
quarterly_data = quarterly_data.merge(segment_baselines, on='segment_id', how='left')
quarterly_data['segment_std_accidents'] = quarterly_data['segment_std_accidents'].fillna(0)

print(" Calculated lag features")

# ============================================================================
# STEP 5: CREATE PREDICTION DATAFRAME FOR NEXT QUARTER
# ============================================================================


print("STEP 5: PREPARING NEXT QUARTER PREDICTION DATA")


# Get unique segments
segments = sorted(quarterly_data['28milename'].unique())
next_quarter_data = []

for segment_name in segments:
    segment_id = segment_map[segment_name]
    
    # Get most recent data for this segment
    segment_history = quarterly_data[quarterly_data['28milename'] == segment_name].sort_values(
        ['CRASH_YEAR', 'quarter_num']
    )
    
    if len(segment_history) == 0:
        continue
    
    latest = segment_history.iloc[-1]
    
    # Get lag features from most recent data
    accidents_lag1 = latest['total_accidents']  # Most recent quarter becomes lag1
    
    # Get lag4 (same quarter last year) - need to look back 4 quarters
    if len(segment_history) >= 4:
        accidents_lag4 = segment_history.iloc[-4]['total_accidents']
    else:
        accidents_lag4 = latest['accidents_lag4']
    
    # Calculate rolling mean from last 4 quarters
    if len(segment_history) >= 4:
        accidents_rolling_mean = segment_history.tail(4)['total_accidents'].mean()
        accidents_rolling_std = segment_history.tail(4)['total_accidents'].std()
    else:
        accidents_rolling_mean = latest['accidents_rolling_mean']
        accidents_rolling_std = latest['accidents_rolling_std']
    
    # Get segment baseline statistics
    segment_mean = latest['segment_mean_accidents']
    segment_std = latest['segment_std_accidents']
    
    # Get recent severe accidents and other features
    recent_severe = segment_history.tail(4)['severe_accidents'].mean()
    recent_poor_road = segment_history.tail(4)['poor_road_pct'].mean()
    recent_collision_div = segment_history.tail(4)['collision_type_diversity'].mean()
    
    # Get coordinates
    lat = latest['14milelat']
    lon = latest['14milelon']
    
    next_quarter_data.append({
        '28milename': segment_name,
        '14milelat': lat,
        '14milelon': lon,
        'CRASH_YEAR': next_year,
        'Quarter': next_quarter,
        'quarter_num': next_quarter_num,
        'is_winter': 1 if next_quarter_num in [1, 4] else 0,
        'segment_id': segment_id,
        'accidents_lag1': accidents_lag1,
        'accidents_lag4': accidents_lag4,
        'accidents_rolling_mean': accidents_rolling_mean,
        'accidents_rolling_std': accidents_rolling_std if not np.isnan(accidents_rolling_std) else 0,
        'segment_mean_accidents': segment_mean,
        'segment_std_accidents': segment_std,
        'severe_accidents': recent_severe,
        'poor_road_pct': recent_poor_road,
        'collision_type_diversity': recent_collision_div
    })

prediction_df = pd.DataFrame(next_quarter_data)
print(f" Created prediction data for {len(prediction_df)} segments")

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================


print("STEP 6: MAKING PREDICTIONS")


# Prepare features
X_predict = prediction_df[feature_cols]

# Fill any remaining NaN values
X_predict = X_predict.fillna(X_predict.median())

# Make predictions for total accidents
predicted_accidents = model.predict(X_predict)
predicted_accidents = np.maximum(predicted_accidents, 0).round().astype(int)

prediction_df['predicted_total_accidents'] = predicted_accidents

print(" Predicted total accidents")

# ============================================================================
# STEP 7: PREDICT SEVERE ACCIDENTS
# ============================================================================


print("STEP 7: PREDICTING SEVERE ACCIDENTS")


# Calculate historical severe accident ratio for each segment
severe_ratios = quarterly_data.groupby('segment_id').apply(
    lambda x: x['severe_accidents'].sum() / x['total_accidents'].sum() if x['total_accidents'].sum() > 0 else 0
).to_dict()

# Apply ratios to predictions
prediction_df['severe_accident_ratio'] = prediction_df['segment_id'].map(severe_ratios)
prediction_df['predicted_severe_accidents'] = (
    prediction_df['predicted_total_accidents'] * prediction_df['severe_accident_ratio']
).round().astype(int)

# Ensure severe accidents don't exceed total
prediction_df['predicted_severe_accidents'] = prediction_df[['predicted_severe_accidents', 'predicted_total_accidents']].min(axis=1)

print(" Predicted severe accidents")

# ============================================================================
# STEP 8: CREATE FINAL OUTPUT
# ============================================================================


print("STEP 8: CREATING OUTPUT FILE")


# Create final output with requested columns
output_df = prediction_df[[
    '28milename',
    '14milelat',
    '14milelon',
    'predicted_total_accidents',
    'predicted_severe_accidents'
]].copy()

output_df.columns = [
    '28milename',
    '14milelat',
    '14milelon',
    'predicted_total_accidents',
    'predicted_severe_accidents'
]

# Sort by segment name
output_df = output_df.sort_values('28milename')

# Save to CSV
optimizedf = output_df.rename(columns={'predicted_total_accidents': 'CRN', 'predicted_severe_accidents': 'is_severe'}) #to ensure file naming consistency
script_dir = Path(__file__).parent
filename = script_dir.parent / 'Data' / f'predictions_{next_quarter}_{next_year}.csv'
optimizedf.to_csv(filename, index=False)
print(f"Saved {filename}")

# ============================================================================
# STEP 9: DISPLAY RESULTS
# ============================================================================


print(f"PREDICTIONS FOR {next_quarter} {next_year}")


print("\n" + output_df.to_string(index=False))


print("SUMMARY STATISTICS")


print(f"\nTotal predicted accidents: {output_df['predicted_total_accidents'].sum()}")
print(f"Total predicted severe accidents: {output_df['predicted_severe_accidents'].sum()}")
print(f"Average per segment: {output_df['predicted_total_accidents'].mean():.1f} accidents")
print(f"Severe accident rate: {output_df['predicted_severe_accidents'].sum() / output_df['predicted_total_accidents'].sum() * 100:.1f}%")

print("\nHIGHEST RISK SEGMENTS:")
top_segments = output_df.nlargest(5, 'predicted_total_accidents')[['28milename', 'predicted_total_accidents', 'predicted_severe_accidents']]
print(top_segments.to_string(index=False))

print("\nSEGMENTS WITH MOST SEVERE ACCIDENTS:")
top_severe = output_df.nlargest(5, 'predicted_severe_accidents')[['28milename', 'predicted_total_accidents', 'predicted_severe_accidents']]
print(top_severe.to_string(index=False))


print("PREDICTION COMPLETE!")

print(f" Output file: {filename}")
print(" File contains: 28milename, 14milelat, 14milelon, predicted_total_accidents (CRN), predicted_severe_accidents (is_severe)")