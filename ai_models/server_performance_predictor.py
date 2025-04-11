import pandas as pd
import numpy as np
import joblib
import os
import sqlite3

def load_model_components(model_dir="./server_performance_model"):
    """Load the trained model, scaler and feature list."""
    model = joblib.load(os.path.join(model_dir, "performance_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))
    return model, scaler, features

def predict_degradation(temperature, humidity, server_load, fans_speed, model_dir="./server_performance_model"):
    """Predict server degradation from temperature log data."""
    # Load model components
    model, scaler, features = load_model_components(model_dir)
    
    # Create a mapping from available data to expected features
    # We'll need to make some assumptions for missing features
    feature_data = {
        # Map available fields to their closest corresponding features
        'load-1m': server_load,                # Use server_load as the 1-minute load
        'load-5m': server_load * 0.9,          # Estimate 5-minute load as slightly lower
        'load-15m': server_load * 0.8,         # Estimate 15-minute load as even lower
        'sys-thermal': temperature,            # Use temperature as thermal data
        
        # Set defaults for unavailable data with reasonable values
        'mem_usage_pct': 0.7,                  # Assume 70% memory usage
        'swap_usage_pct': 0.3,                 # Assume 30% swap usage
        'cpu-iowait': 1.0,                     # Assume 1% CPU in I/O wait
        'cpu-system': 5.0,                     # Assume 5% CPU in system tasks
        'cpu-user': 30.0                       # Assume 30% CPU in user tasks
    }
    
    # Calculate load_delta like in the training code
    feature_data['load_delta'] = feature_data['load-1m'] - feature_data['load-5m']
    
    # Create a DataFrame with the expected features in the correct order
    input_df = pd.DataFrame([feature_data])
    
    # Extract just the features the model expects
    input_features = input_df[features]
    
    # Scale the features using the same scaler used during training
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    degradation_probability = model.predict_proba(input_scaled)[0, 1]
    degradation_predicted = model.predict(input_scaled)[0]
    
    return {
        'degradation_predicted': bool(degradation_predicted),
        'degradation_probability': float(degradation_probability),
        'status': 'Potential degradation detected' if degradation_predicted else 'Normal operation'
    }

# Example of predicting from database records
def predict_from_recent_logs(db_path, num_records=10, model_dir="./server_performance_model"):
    """Get predictions for the most recent records in the database."""
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the most recent records
    query = """
    SELECT timestamp, temperature, humidity, server_load, fans_speed
    FROM temperature_logs
    ORDER BY timestamp DESC
    LIMIT ?
    """
    cursor.execute(query, (num_records,))
    records = cursor.fetchall()
    
    # Make predictions for each record
    results = []
    for record in records:
        timestamp, temperature, humidity, server_load, fans_speed = record
        # Use 0 as default for NULL values
        humidity = humidity or 0
        server_load = server_load or 0
        fans_speed = fans_speed or 0
        
        prediction = predict_degradation(temperature, humidity, server_load, fans_speed, model_dir)
        results.append({
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'server_load': server_load,
            'fans_speed': fans_speed,
            'prediction': prediction
        })
    
    conn.close()
    return results