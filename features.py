import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
    """
    Load raw energy data and resample to hourly intervals.
    Handles missing values and creates a clean timeseries.
    """
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        low_memory=False
    )
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.set_index('datetime')
    
    # We are primarily interested in Global_active_power for energy_consumption target
    df = df[['Global_active_power']]
    
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    
    # Resample to hourly average first
    hourly_df = df.resample('h').mean()
    
    # Rename target column to match expected name
    hourly_df.rename(columns={'Global_active_power': 'energy_consumption'}, inplace=True)
    
    # Forward fill/interpolate small gaps up to 2 hours. 
    # Long gaps (up to 5 days exist) will remain NaN.
    # This ensures that rolling windows and lags over large gaps result in NaNs, 
    # which we will safely drop before training to avoid distortion.
    hourly_df['energy_consumption'] = hourly_df['energy_consumption'].interpolate(method='linear', limit=2)
    
    return hourly_df


def create_features(history: pd.Series, timestamps: pd.DatetimeIndex, is_training: bool = False):
    """
    Generate time-series features.
    
    Parameters:
    - history: pandas Series of 'energy_consumption'.
    - timestamps: The timestamps for which we want to generate features.
                  During training, this is typically the entire history.index.
                  During inference, this is a single timestamp or small range.
    - is_training: Boolean flag. If True, computes features efficiently over the whole series 
                   and STRICTLY shifts to avoid target leakage. 
                   If False, computes features assuming 'history' contains past data up to t-1.
    """
    if is_training:
        # In training, history contains ALL data including targets.
        # We must shift by 1 so that features at time t only see data up to t-1.
        past_data = history.shift(1)
        # Create a DataFrame to hold features
        features = pd.DataFrame(index=history.index)
        t_index = history.index
    else:
        # In inference, history only contains data up to t-1.
        # We generate features for the given timestamps
        past_data = history.copy()
        
        # Ensure timestamps is a DatetimeIndex
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
            
        # We create a dummy DataFrame to compute the rolling features over history,
        # then we will extract the specific timestamps we need
        features = pd.DataFrame(index=timestamps)
        t_index = timestamps

    # Date/time features (these don't leak future target info)
    features["hour"] = t_index.hour
    features["day_of_week"] = t_index.dayofweek
    features["month"] = t_index.month
    features["day"] = t_index.day
    features["year"] = t_index.year

    features["is_weekend"] = (t_index.dayofweek >= 5).astype(int)
    features["is_morning"] = ((t_index.hour >= 6) & (t_index.hour < 12)).astype(int)
    features["is_afternoon"] = ((t_index.hour >= 12) & (t_index.hour < 18)).astype(int)
    features["is_evening"] = ((t_index.hour >= 18) & (t_index.hour < 23)).astype(int)
    features["is_night"] = ((t_index.hour >= 23) | (t_index.hour < 6)).astype(int)

    # Cyclical features
    features["hour_sin"] = np.sin(2 * np.pi * t_index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * t_index.hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * t_index.dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * t_index.dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * t_index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * t_index.month / 12)

    # Lag features
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
    
    if is_training:
        for lag in lags:
            features[f"lag_{lag}"] = past_data.shift(lag - 1)
            
        windows = [6, 12, 24, 48, 72, 168]
        for window in windows:
            rolling_obj = past_data.rolling(window=window)
            features[f"rolling_mean_{window}"] = rolling_obj.mean()
            features[f"rolling_std_{window}"] = rolling_obj.std()
            features[f"rolling_min_{window}"] = rolling_obj.min()
            features[f"rolling_max_{window}"] = rolling_obj.max()
            
        features["diff_1"] = past_data - past_data.shift(1)
        features["diff_24"] = past_data - past_data.shift(24)
        features["diff_168"] = past_data - past_data.shift(168)
        
    else:
        # Inference mode: calculate features for specific timestamps
        # assuming past_data contains everything up to t-1 for the FIRST timestamp.
        # If timestamps has length > 1, this logic needs to be iterative, but typically
        # we call this one step at a time for recursive forecasting.
        
        for t in timestamps:
            for lag in lags:
                if len(past_data) >= lag:
                    features.loc[t, f"lag_{lag}"] = past_data.iloc[-lag]
                else:
                    features.loc[t, f"lag_{lag}"] = np.nan
                    
            windows = [6, 12, 24, 48, 72, 168]
            for window in windows:
                if len(past_data) >= window:
                    values = past_data.iloc[-window:]
                    features.loc[t, f"rolling_mean_{window}"] = values.mean()
                    features.loc[t, f"rolling_std_{window}"] = values.std()
                    features.loc[t, f"rolling_min_{window}"] = values.min()
                    features.loc[t, f"rolling_max_{window}"] = values.max()
                else:
                    features.loc[t, f"rolling_mean_{window}"] = np.nan
                    features.loc[t, f"rolling_std_{window}"] = np.nan
                    features.loc[t, f"rolling_min_{window}"] = np.nan
                    features.loc[t, f"rolling_max_{window}"] = np.nan
                    
            if len(past_data) >= 2:
                features.loc[t, "diff_1"] = past_data.iloc[-1] - past_data.iloc[-2]
            else:
                features.loc[t, "diff_1"] = np.nan
                
            if len(past_data) >= 25:
                features.loc[t, "diff_24"] = past_data.iloc[-1] - past_data.iloc[-25]
            else:
                features.loc[t, "diff_24"] = np.nan
                
            if len(past_data) >= 169:
                features.loc[t, "diff_168"] = past_data.iloc[-1] - past_data.iloc[-169]
            else:
                features.loc[t, "diff_168"] = np.nan

    return features

