# import os
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import joblib
# import json
# from datetime import datetime
# import sqlite3

# class TemperaturePredictor:
#     def __init__(self, model_dir='models'):
#         """Initialize the predictor with the trained model files."""
#         # Load the XGBoost model
#         model_path = os.path.join(model_dir, 'xgboost_temperature_model.json')
#         print(f"Loading model from {model_path}...")
#         self.model = xgb.XGBRegressor()
#         self.model.load_model(model_path)
        
#         # Load the scaler
#         scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
#         self.scaler = joblib.load(scaler_path)
        
#         # Load feature names
#         with open(os.path.join(model_dir, 'feature_names.json'), 'r') as f:
#             self.feature_names = json.load(f)
        
#         # Load metadata
#         with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
#             self.metadata = json.load(f)
        
#         print(f"Model loaded successfully - created on {self.metadata.get('creation_date')}")
    
#     def prepare_features_from_db_row(self, row):
#         """
#         Prepare features from database row format.
        
#         Args:
#             row: A dictionary or namedtuple with fields matching the temperature_logs table
            
#         Returns:
#             DataFrame with properly formatted features
#         """
#         # Parse timestamp to datetime
#         if isinstance(row.get('timestamp', ''), str):
#             dt = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
#         elif isinstance(row.get('timestamp'), datetime):
#             dt = row['timestamp']
#         else:
#             # Default to current time if timestamp is missing or invalid
#             dt = datetime.now()
        
#         # Create base features dictionary
#         features = {
#             'humidity': float(row.get('humidity', 50.0)),
#             'hour': dt.hour,
#             'day': dt.day,
#             'month': dt.month,
#             'day_of_week': dt.weekday(),
#             'is_weekend': 1 if dt.weekday() >= 5 else 0,
#         }
        
#         # Add cyclical encoding for time features
#         features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24.0)
#         features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24.0)
#         features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12.0)
#         features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12.0)
#         features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7.0)
#         features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7.0)
        
#         # Add server metrics if available
#         if 'server_load' in row:
#             features['server_load'] = float(row.get('server_load', 0.0))
#         if 'fans_speed' in row:
#             features['fans_speed'] = float(row.get('fans_speed', 0.0))
            
#         # Create a single-row DataFrame
#         df = pd.DataFrame([features])
        
#         return df
    
#     def predict(self, input_data):
#         """
#         Make a prediction using the model.
        
#         Args:
#             input_data: DataFrame or dict with input features
            
#         Returns:
#             Predicted temperature value
#         """
#         # If input is a dict (single record), convert to DataFrame
#         if isinstance(input_data, dict):
#             X = self.prepare_features_from_db_row(input_data)
#         else:
#             X = input_data.copy()
        
#         # Ensure all required features are present
#         for feature in self.feature_names:
#             if feature not in X.columns:
#                 X[feature] = 0  # Default value for missing features
        
#         # Keep only relevant features in the right order
#         X = X[self.feature_names]
        
#         # Scale features
#         X_scaled = self.scaler.transform(X)
        
#         # Make prediction
#         prediction = self.model.predict(X_scaled)[0]
        
#         return prediction

#     def predict_from_database(self, db_path, limit=1):
#         """
#         Make predictions using the latest data from the database.
        
#         Args:
#             db_path: Path to SQLite database
#             limit: Number of most recent records to use
            
#         Returns:
#             Dictionary with predictions and input data
#         """
#         conn = sqlite3.connect(db_path)
#         conn.row_factory = sqlite3.Row
#         cursor = conn.cursor()
        
#         # Get the most recent records
#         cursor.execute("""
#             SELECT * FROM temperature_logs
#             ORDER BY id DESC
#             LIMIT ?
#         """, (limit,))
        
#         rows = cursor.fetchall()
#         conn.close()
        
#         if not rows:
#             return {"error": "No data found in database"}
        
#         results = []
#         for row in rows:
#             # Convert Row to dict
#             data = {key: row[key] for key in row.keys()}
            
#             # Make prediction
#             predicted_temp = self.predict(data)
            
#             # Add prediction to results
#             data['predicted_temperature'] = round(float(predicted_temp), 2)
#             results.append(data)
            
#         return results[0] if limit == 1 else results


# # Example usage
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Predict temperature using trained XGBoost model")
#     parser.add_argument("--db", type=str, help="Path to SQLite database")
#     parser.add_argument("--timestamp", type=str, help="Timestamp in YYYY-MM-DD HH:MM:SS format")
#     parser.add_argument("--humidity", type=float, default=50.0, help="Current humidity")
#     parser.add_argument("--server_load", type=float, default=0.0, help="Server load")
#     parser.add_argument("--fans_speed", type=float, default=0.0, help="Fans speed")
    
#     args = parser.parse_args()
    
#     # Initialize predictor
#     predictor = TemperaturePredictor()
    
#     # If database path provided, predict from database
#     if args.db:
#         print(f"Getting data from database: {args.db}")
#         result = predictor.predict_from_database(args.db)
#         print(f"\nPrediction using latest database record:")
#         print(f"Timestamp: {result.get('timestamp')}")
#         print(f"Current humidity: {result.get('humidity')}%")
#         print(f"Server load: {result.get('server_load')}")
#         print(f"Fans speed: {result.get('fans_speed')}")
#         print(f"Predicted temperature: {result.get('predicted_temperature')}°C")
#     else:
#         # Use command line arguments or defaults
#         timestamp = args.timestamp or "2025-04-10 16:18:04"
        
#         # Create input data
#         input_data = {
#             'timestamp': timestamp,
#             'humidity': args.humidity,
#             'server_load': args.server_load,
#             'fans_speed': args.fans_speed
#         }
        
#         print(f"\nInput data:")
#         for key, value in input_data.items():
#             print(f"{key}: {value}")
        
#         # Make prediction
#         predicted_temp = predictor.predict(input_data)
        
#         print(f"\nPredicted temperature: {predicted_temp:.2f}°C")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

def predict_using_trained_model(recent_data_path, model_folder, prediction_hours=1):
    """
    Use the pre-trained model to predict temperature
    
    Parameters:
    recent_data_path: Path to CSV file with recent server room data
    model_folder: Folder containing the saved model files
    prediction_hours: Number of hours ahead to predict
    
    Returns:
    Predicted temperature
    """
    print(f"Looking for model in: {os.path.abspath(model_folder)}")
    
    # Check if model files exist
    model_path = os.path.join(model_folder, 'xgb_model.pkl')
    features_path = os.path.join(model_folder, 'feature_columns.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature columns file not found at {features_path}")
        
    # Load the model
    xgb_model = joblib.load(model_path)
    
    # Load feature columns
    feature_columns = joblib.load(features_path)
    print(f"Loaded model with {len(feature_columns)} features")
    
    # Load and preprocess recent data
    print(f"Loading data from: {recent_data_path}")
    recent_data = pd.read_csv(recent_data_path)
    print(f"Loaded data with {len(recent_data)} rows and columns: {list(recent_data.columns)}")
    
    processed_data = preprocess_data(recent_data)
    
    # Prepare features for prediction
    X_pred = prepare_prediction_features(processed_data, feature_columns)
    
    # Make prediction
    prediction = xgb_model.predict(X_pred.iloc[-1:].values)[0]
    
    # Get timestamp for prediction
    last_timestamp = processed_data['datetime'].iloc[-1]
    prediction_timestamp = last_timestamp + timedelta(hours=prediction_hours)
    
    print(f"\nPrediction Results for {prediction_timestamp}:")
    print(f"Temperature Prediction: {prediction:.2f}°C")
    
    return {
        'timestamp': prediction_timestamp,
        'prediction': prediction
    }

def preprocess_data(df):
    """
    Preprocess the data for prediction, similar to the training preprocessing
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert datetime format
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')
    
    # Sort by datetime
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # Extract time-based features
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['day_of_month'] = data['datetime'].dt.day
    data['month'] = data['datetime'].dt.month
    
    # Create cyclical features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week']/7)
    data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week']/7)
    
    # Create lag features for temperature
    for lag in [1, 2, 3, 6, 12, 24]:
        if len(data) > lag:
            data[f'temp_lag_{lag}'] = data['temperature'].shift(lag)
    
    # Calculate rate of change
    if len(data) > 1:
        data['temp_diff'] = data['temperature'].diff()
        data['temp_diff_rate'] = data['temp_diff'] / data['temperature'].shift(1) * 100
    
    # Calculate rolling statistics
    if len(data) > 6:
        data['temp_rolling_mean_6'] = data['temperature'].rolling(window=6).mean()
        data['temp_rolling_std_6'] = data['temperature'].rolling(window=6).std()
    
    # Drop rows with NaN values
    data_clean = data.dropna().reset_index(drop=True)
    
    return data_clean

def prepare_prediction_features(data, feature_columns):
    """
    Prepare the features needed for prediction
    """
    # Select all existing columns that match the expected feature columns
    available_features = [col for col in feature_columns if col in data.columns]
    X = data[available_features].copy()
    
    # Add any missing columns with zeros
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    
    # Ensure columns are in the correct order
    X = X[feature_columns]
    
    # Handle any NaN or infinite values
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    return X

if __name__ == "__main__":
    print("Temperature Prediction Using Pre-trained Model")
    print(f"Current Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model folder is found correctly
    model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Properly get user input for data file
    print("\nPlease enter the path to your CSV file with recent data:")
    recent_data_path = input("> ").strip()
    
    # Validate the file exists
    while not os.path.isfile(recent_data_path):
        print(f"Error: File not found at {recent_data_path}")
        print("Please enter a valid path to your CSV file (or 'exit' to quit):")
        recent_data_path = input("> ").strip()
        if recent_data_path.lower() == 'exit':
            print("Exiting program.")
            exit()
    
    # How many hours ahead to predict
    print("\nHow many hours ahead would you like to predict? (default: 1)")
    try:
        prediction_hours = int(input("> ") or "1")
    except ValueError:
        print("Invalid input, using default of 1 hour")
        prediction_hours = 1
    
    # Make prediction
    try:
        prediction = predict_using_trained_model(
            recent_data_path=recent_data_path,
            model_folder=model_folder,
            prediction_hours=prediction_hours
        )
        
        print("\nSuccessfully generated prediction!")
        print(f"Predicted temperature for {prediction['timestamp']}: {prediction['prediction']:.2f}°C")
        
    except Exception as e:
        print(f"\nError during prediction: {e}")