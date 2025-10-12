```markdown
# EMS Vehicle Staging Location Optimization on I-76 in Pennsylvania

This project employs a comprehensive machine learning and optimization framework to determine optimal Emergency Medical Services (EMS) vehicle staging locations along the I-76 corridor in Pennsylvania. The methodology integrates predictive modeling of traffic accident patterns with Mixed-Integer Linear Programming (MILP) to minimize emergency response times to anticipated crash hotspots.

---

## Project Overview

The analysis is conducted in three sequential phases:

1. **Predictive Model Training**: Development and validation of machine learning models to forecast quarterly accident occurrences across highway segments.
2. **Accident Prediction**: Generation of segment-level accident predictions for future quarters using the trained models.
3. **Optimization Analysis**: Determination of optimal EMS staging locations through various MILP formulations that minimize response distances to predicted crash hotspots.

---

## Quick Start Guide

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Jupyter Notebook (for optimization analysis)

### Installation and Execution

Follow these steps to execute the complete analysis pipeline:

#### Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pulp osmnx geopandas folium geopy
```


#### Step 2: Verify Data Files

Ensure the following data files are in place:
- `Prediction model/full_data_v2.csv`
- `Data/i76_exits.csv`

#### Step 3: Train the Prediction Model

Go to the Notebook folder, and execute the provided jupyter notebook.

**Expected Output:**
- Model training progress and performance metrics
- Generated files: `accident_prediction_model.pkl`, `model_features.pkl`, `segment_mapping.pkl`
- Visualization: `model_performance.png`
- Validation results: `quarterly_predictions_2024.csv`

#### Step 4: Generate Predictions for Next Quarter

```bash
python quarter_prediction.py
```

**Expected Output:**
- Prediction progress for each step
- Generated file: `../Data/predictions_{Quarter}_{Year}.csv`
- Console output showing:
  - Predicted quarter and year
  - Summary statistics
  - Highest risk segments
  - Segments with most severe accidents

#### Step 5: Run Optimization Analysis

Run the Final Optimization file present in Notebooks folder.

Open and execute the optimization notebooks sequentially:
1. Open the first optimization notebook
2. Run all cells (Cell → Run All)
3. Review generated maps and optimization results
4. Repeat for remaining notebooks

**Expected Output:**
- Interactive HTML maps: `final_severity_locations_map.html`
- Optimal staging locations printed to console
- Objective function values and optimization statistics

### Verification

Upon successful execution, you should have:

✅ Trained model files (`.pkl` format) in `Prediction model/` folder  
✅ Performance visualization (`model_performance.png`)  
✅ Prediction file in `Data/` folder  
✅ Interactive HTML maps in `Notebooks/` folder  
✅ Console output showing optimal EMS staging locations

---

## Data Requirements

The analysis utilizes the following data files:

### Input Data

1. **`full_data_v2.csv`**: Historical crash data repository containing:
   - `CRN`: Crash Record Number (unique identifier)
   - `28milename`: Highway segment identifier (0.28-mile segments)
   - `14milelat`, `14milelon`: Latitude and longitude coordinates at 0.14-mile resolution
   - `CRASH_YEAR`: Year of crash occurrence
   - `CRASH_MONTH`: Month of crash occurrence
   - `MAX_SEVERITY_LEVEL`: Severity classification (1-5, where 1-2 represent severe crashes)
   - `FATAL_COUNT`: Number of fatalities
   - `TOT_INJ_COUNT`: Total injury count
   - `ROAD_CONDITION`: Road surface condition at time of crash (1-4, where 2-4 represent poor conditions)
   - `COLLISION_TYPE`: Type of collision

2. **`i76_exits.csv`**: Potential EMS staging locations along I-76, comprising:
   - `Latitude`, `Longitude`: Geographic coordinates of exit locations
   - `OSM_ID`: OpenStreetMap unique identifier
   - `Exit_Number`: Highway exit designation
   - `Name`: Exit name or location descriptor

### Generated Data

3. **`predictions_{Quarter}_{Year}.csv`**: Model-generated predictions containing:
   - `28milename`: Highway segment identifier
   - `14milelat`, `14milelon`: Segment coordinates
   - `CRN`: Predicted number of crashes (renamed from `predicted_total_accidents`)
   - `is_severe`: Predicted count of severe crashes (renamed from `predicted_severe_accidents`)

---

## Methodology

### Phase 1: Predictive Model Development (`prediction_model.py`)

This module implements a comprehensive machine learning pipeline for accident prediction:

**Feature Engineering:**
- Temporal features: quarterly aggregation, seasonal indicators (winter vs. non-winter)
- Spatial features: segment-specific identifiers and coordinates
- Lag features: previous quarter accidents, same quarter previous year, 4-quarter rolling statistics
- Historical baselines: segment-level mean and standard deviation
- Crash characteristics: severe accident counts, road condition metrics, collision type diversity

**Model Development:**
- **Base Model**: Random Forest Regressor with conservative hyperparameters to prevent overfitting
- **Enhanced Model**: Gradient Boosting Regressor with cross-validated parameter selection
- **Benchmark Model**: Ridge Regression for linear baseline comparison

**Validation:**
- Temporal train-test split (2022-2023 for training, 2024 for testing)
- 3-fold cross-validation for hyperparameter optimization
- Performance metrics: RMSE, MAE, R²

**Outputs:**
- `accident_prediction_model.pkl`: Serialized trained model
- `model_features.pkl`: Feature column specifications
- `segment_mapping.pkl`: Segment identifier mapping
- `model_performance.png`: Visualization of model performance
- `quarterly_predictions_2024.csv`: Validation set predictions

### Phase 2: Future Quarter Prediction (`quarter_prediction.py`)

This module applies the trained model to generate forward-looking predictions:

**Process:**
1. Load trained model artifacts and historical data
2. Identify most recent quarter in dataset
3. Calculate next quarter to predict
4. Aggregate historical data and compute lag features for prediction horizon
5. Generate predictions for all highway segments
6. Estimate severe accident counts using historical severity ratios
7. Export predictions in optimization-ready format

**Output:**
- `predictions_{Quarter}_{Year}.csv`: Quarterly predictions for all segments

### Phase 3: Optimization Analysis (Notebooks)

The optimization phase implements multiple MILP formulations to determine optimal EMS staging strategies:

#### 1. Initial Model (Minimizing Maximum Distance)

This model identifies a single optimal ambulance location that minimizes the maximum travel distance to any crash hotspot, ensuring equitable coverage across all segments. This minimax formulation guarantees that no hotspot is excessively remote from emergency services.

#### 2. Weighted Optimization Model (by Crash Count)

This formulation determines a single optimal location by minimizing the total weighted travel distance, where weights correspond to predicted crash frequencies (`CRN`). This approach prioritizes high-volume crash locations.

#### 3. Weighted Optimization Model (by Severity)

Similar to the crash count model, this formulation weights hotspots by predicted severe crash counts (`is_severe`), thereby prioritizing locations where serious incidents are anticipated and response time is most critical.

#### 4. Multi-Vehicle Ambulance Staging Model

This advanced model determines optimal locations for 13 ambulances with the constraint that exactly one ambulance is positioned within each of 13 predefined highway stretches. The objective function minimizes total severity-weighted response distance across all hotspots, ensuring comprehensive coverage while accounting for incident severity.

---

## Reproduction Steps

To replicate this analysis, follow these sequential steps:

### 1. Environment Setup

Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pulp osmnx geopandas folium geopy
```

### 2. Data Preparation

Ensure the following data files are present in the appropriate directories:
- `full_data_v2.csv` in the `Prediction model` folder
- `i76_exits.csv` in the `Data` folder

### 3. Model Training

Navigate to the Prediction model folder and execute:

```bash
cd "Prediction model"
python prediction_model.py
```

This script will:
- Load and preprocess historical crash data
- Engineer temporal and spatial features
- Train and validate multiple predictive models
- Save trained model artifacts (`.pkl` files)
- Generate performance visualizations and validation results

### 4. Prediction Generation

In the same directory, execute:

```bash
python quarter_prediction.py
```

This script will:
- Load the trained model and supporting artifacts
- Determine the next quarter to predict
- Generate segment-level predictions for total and severe accidents
- Save predictions to `Data/predictions_{Quarter}_{Year}.csv`

### 5. Optimization Analysis

Navigate to the Notebooks folder and execute the optimization notebooks sequentially:

```bash
cd ../Notebooks
```

Open and run the Jupyter notebooks in order. These notebooks will:
- Load prediction data and staging location candidates
- Formulate and solve various MILP optimization problems
- Generate interactive visualizations of optimal staging configurations
- Export results as HTML map files

---

## Output Artifacts

### Predictive Modeling Outputs

**Model Files:**
- `accident_prediction_model.pkl`: Trained prediction model
- `model_features.pkl`: Feature specifications
- `segment_mapping.pkl`: Segment identifier mappings

**Validation Results:**
- `model_performance.png`: Comparative visualization of model performance
- `quarterly_predictions_2024.csv`: Detailed validation predictions with error metrics

### Prediction Outputs

- `predictions_{Quarter}_{Year}.csv`: Segment-level predictions for the specified quarter

### Optimization Outputs

For each optimization model, the following information is generated:

**Console Output:**
- Optimization status (e.g., "Optimal")
- Selected staging location(s) with complete metadata (OSM_ID, name, exit number, coordinates)
- Objective function value (e.g., minimized maximum distance, total weighted distance)

**Visualization:**
- `final_severity_locations_map.html`: Interactive map displaying:
  - I-76 highway corridor
  - All candidate staging locations (exit points)
  - Crash hotspots with proportional markers indicating predicted crash counts or severity
  - Optimal staging location(s) denoted with star icons

---

## Dependencies

### Core Data Science Libraries
- [**pandas**](https://pandas.pydata.org/): Data manipulation and analysis
- [**numpy**](https://numpy.org/): Numerical computing
- [**matplotlib**](https://matplotlib.org/): Visualization and plotting
- [**seaborn**](https://seaborn.pydata.org/): Statistical data visualization
- [**scikit-learn**](https://scikit-learn.org/): Machine learning algorithms and utilities

### Optimization and Geospatial Libraries
- [**pulp**](https://coin-or.github.io/pulp/): Linear programming and optimization
- [**osmnx**](https://osmnx.readthedocs.io/): OpenStreetMap network analysis
- [**geopandas**](https://geopandas.org/): Geospatial data operations
- [**folium**](https://python-visualization.github.io/folium/): Interactive mapping
- [**geopy**](https://geopy.readthedocs.io/): Geocoding and distance calculations

### Python Standard Library
- **pickle**: Object serialization
- **warnings**: Warning control
- **pathlib**: Filesystem path operations

---

## Project Structure

```
.
├── Prediction model/
│   ├── prediction_model.py          # Model training script
│   ├── quarter_prediction.py        # Prediction generation script
│   ├── full_data_v2.csv            # Historical crash data (input)
│   ├── accident_prediction_model.pkl # Trained model (output)
│   ├── model_features.pkl           # Feature specifications (output)
│   ├── segment_mapping.pkl          # Segment mappings (output)
│   ├── model_performance.png        # Performance visualization (output)
│   └── quarterly_predictions_2024.csv # Validation results (output)
├── Data/
│   ├── i76_exits.csv               # Staging location candidates (input)
│   └── predictions_{Q}_{Year}.csv  # Generated predictions (output)
├── Notebooks/
│   └── *.ipynb                      # Optimization notebooks
└── README.md
```

---

## Technical Notes

- The analysis explicitly excludes the `mile344` segment from all computations due to data quality concerns.
- Model training employs conservative hyperparameters to mitigate overfitting and enhance generalizability to future quarters.
- Lag features utilize segment-specific imputation strategies to handle missing historical data for newer or low-activity segments.
- The optimization framework accommodates multiple objective functions and constraints to address varying operational priorities and resource availability scenarios.

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: `FileNotFoundError` for data files
- **Solution**: Verify data files are in correct directories as specified in Project Structure

**Issue**: Memory errors during model training
- **Solution**: Reduce `n_estimators` parameter in `prediction_model.py` or use a machine with more RAM

**Issue**: Jupyter notebook kernel crashes during optimization
- **Solution**: Restart kernel and run cells individually rather than "Run All"

---

## License

[Specify your license here]

---

## Citation

When utilizing this methodology or framework, please acknowledge the integrated machine learning and optimization approach for emergency services resource allocation along highway corridors.

---

## Contact

[Your contact information or contribution guidelines]

---

## Acknowledgments

[Acknowledge data sources, collaborators, or funding sources if applicable]
```