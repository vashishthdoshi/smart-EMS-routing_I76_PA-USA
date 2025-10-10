# EMS Vehicle Staging Location Optimization on I-76 in Pennsylvania

This project uses Mixed-Integer Linear Programming (MILP) to find the best locations to stage Emergency Medical Services (EMS) vehicles along the I-76 corridor in Pennsylvania. The goal is to minimize response times to crash hotspots by analyzing crash data and potential staging locations.

---

## Data Required

To run this analysis, you will need the following CSV files:

1.  **`predictions_Q1_2025.csv`**: This file should contain information about crash hotspots along I-76, including:
    * `14milelat`, `14milelon`: The latitude and longitude of the hotspot.
    * `CRN`: The number of crashes at the hotspot.
    * `is_severe`: A count of severe crashes at the hotspot.

2.  **`i76_exits.csv`**: This file should contain a list of potential ambulance staging locations, which are the exits along I-76. The file should include:
    * `Latitude`, `Longitude`: The coordinates of the exit.
    * `OSM_ID`: A unique identifier for the exit from OpenStreetMap.
    * `Exit_Number`: The exit number.
    * `Name`: The name of the exit.

---

## How it Works

The script builds and solves several optimization models to determine the best EMS staging locations.

### 1. Initial Model (Minimizing Maximum Distance)

This model finds the single best ambulance location to minimize the travel distance to the farthest crash hotspot. This ensures that no single hotspot is excessively far from an ambulance.

### 2. Weighted Optimization Model (by Crash Count)

This model also finds a single optimal location but gives more weight to hotspots with a higher number of crashes (`CRN`). The objective is to minimize the total weighted travel distance.

### 3. Weighted Optimization Model (by Severity)

Similar to the crash count model, this model weights hotspots by the number of severe crashes (`is_severe`). This prioritizes a faster response to more serious incidents.

### 4. Multi-Vehicle Ambulance Staging Model

This is the most advanced model in the script. It is designed to find the optimal locations for 13 ambulances, with the constraint that exactly one ambulance is placed in each of the 13 predefined highway stretches. This model's objective is to minimize the total response distance, weighted by the severity of crashes at each hotspot.

---

## How to Use

1.  **Place Data Files:** Make sure the `predictions_Q1_2025.csv` and `i76_exits.csv` files are in the same directory as the Python script.
2.  **Install Dependencies:** You will need to install the following Python libraries:
    ```bash
    pip install pulp osmnx geopandas folium
    ```
3.  **Run the Script:** Execute the Python script.

---

## Output

The script will print the following to the console for each model:
* The status of the optimization (e.g., "Optimal").
* The optimal staging location(s) with their OSM_ID, name, exit number, and coordinates.
* The minimized objective function value (e.g., "Minimized Maximum Distance").

The script also generates and saves interactive maps as HTML files (`final_severity_locations_map.html`). These maps visualize:
* The I-76 highway.
* All potential staging locations (exits).
* The crash hotspots, with the size of the marker indicating the number or severity of crashes.
* The optimal location(s) found by the models, marked with a star icon.

---

## Dependencies

* [pandas](https://pandas.pydata.org/)
* [geopy](https://geopy.readthedocs.io/en/stable/)
* [folium](https://python-visualization.github.io/folium/)
* [pulp](https://coin-or.github.io/pulp/)
* [osmnx](https://osmnx.readthedocs.io/en/stable/)
* [geopandas](https://geopandas.org/en/stable/)