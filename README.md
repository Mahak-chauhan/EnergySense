# ⚡ EnergySense

## AI-Powered Energy Consumption Forecasting System

EnergySense is a machine learning application designed to forecast household electricity consumption. It uses historical energy patterns and time-series feature engineering to provide multi-step recursive forecasts.

This improved project resolves data leakage issues, introduces strict chronological validation, and utilizes an extremely lightweight and performant `HistGradientBoostingRegressor`.

### 🚀 Features

- **Robust Time-Series Preprocessing:** Resolves target leakage by correctly shifting target variables prior to calculating rolling and lag features.
- **Smart Missing-Data Handling:** Drops training instances disrupted by severe outages (e.g., 5-day gaps) rather than linearly interpolating them, preserving chronological integrity.
- **Efficient ML Model:** Replaced a heavy 75 MB Random Forest with a fast, accurate 568 KB `HistGradientBoostingRegressor`.
- **DRY Architecture:** The exact feature engineering logic (`features.py`) is shared between the training pipeline and the Streamlit app.
- **Recursive Multi-Step Forecasting:** Allows forecasting 1, 6, 12, or 24 hours into the future.

### 📊 Model Performance

The evaluation was performed using a strict chronological holdout (the last 20% of the timeline, roughly 6,400 hours) with a 1-step-ahead prediction strategy.

| Model / Baseline | MAE | RMSE | WAPE | R² Score |
|---|---|---|---|---|
| Persistence Baseline (Predict t-1) | 0.4003 kW | 0.6103 kW | 38.59% | 0.3729 |
| Seasonal Naive (Predict t-24) | 0.5550 kW | 0.8215 kW | 53.50% | -0.1363 |
| **HistGradientBoosting (Candidate)** | **0.3287 kW** | **0.4791 kW** | **31.68%** | **0.6135** |

*Note: Metrics reflect one-step-ahead forecasting on the 6,452-hour blind chronological test holdout (2010-01-10 to 2010-11-26). R² measures the proportion of variance explained (~61.35%), not an "accuracy percentage". WAPE (Weighted Absolute Percentage Error) prevents skew from near-zero power readings. Operational multi-step forecasts (6h, 12h, 24h) use recursive predictions and accumulate uncertainty over time.*

### 📁 Dataset Attribution

The model was trained on the [Individual Household Electric Power Consumption Data Set](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) from the UCI Machine Learning Repository.
* Due to file size limits, the 133 MB raw dataset is **not included** in this repository. 
* A lightweight `recent_history.csv` (containing the last 1000 hours) is provided strictly to seed the Streamlit app's recursive forecasting engine upon startup.

### ⚠️ Project Limitations

- **Recursive Forecasting Uncertainty:** While the model performs very well 1-step ahead, horizons like 24 hours require the model to consume its own predictions as ground truth for lag features. This intrinsically causes errors to compound over time. The UI labels these as "Estimates" to reflect this degradation.

### 🛠️ Local Installation & Execution

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### ☁️ Streamlit Community Cloud Deployment

This project is fully structured for Streamlit Community Cloud. 
1. Push this repository to your GitHub account.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, and set the main file path to `app.py`.
4. Deploy! The app utilizes relative paths and the lightweight `recent_history.csv`, removing any dependency on local data structures.
