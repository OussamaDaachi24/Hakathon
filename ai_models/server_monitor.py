# ai_models/server_monitor.py
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timezone

class ServerHealthMonitor:
    """Server health monitoring system with ML insights and rule-based classification"""
    
    def __init__(self, model_dir="./ai_models/balanced_server_model"):
        """Initialize the monitor with trained model and components"""
        self.model = joblib.load(os.path.join(model_dir, "performance_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.features = joblib.load(os.path.join(model_dir, "features.pkl"))
        
        # Load metadata
        try:
            with open(os.path.join(model_dir, "model_info.txt"), "r") as f:
                self.model_info = f.read()
        except:
            self.model_info = "Model metadata not available"
    
    def predict(self, temperature, server_load, cpu_user,
               cpu_system=0.5, cpu_iowait=0.1, memory_pct=0.2):
        """Predict server health primarily using rule-based classification"""
        # Get current timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create feature data
        feature_data = {
            'load-1m': server_load,
            'load-5m': server_load * 0.9,
            'load-15m': server_load * 0.8,
            'sys-thermal': temperature,
            'mem_usage_pct': memory_pct,
            'swap_usage_pct': 0.05,
            'cpu-iowait': cpu_iowait,
            'cpu-system': cpu_system,
            'cpu-user': cpu_user
        }
        
        # Calculate derived features
        feature_data['load_delta'] = feature_data['load-1m'] - feature_data['load-5m']
        
        if 'cpu_total' in self.features:
            feature_data['cpu_total'] = cpu_user + cpu_system + cpu_iowait
        if 'io_total' in self.features:
            feature_data['io_total'] = 0
        
        # Create DataFrame
        input_df = pd.DataFrame([feature_data])
        input_features = input_df[self.features]
        input_scaled = self.scaler.transform(input_features)
        
        # Get ML prediction probability
        ml_probability = float(self.model.predict_proba(input_scaled)[0, 1])
        
        # RULE-BASED CLASSIFICATION
        # Apply rule-based status determination - entirely separate from ML prediction
        if temperature > 75 or server_load > 3.5 or cpu_user > 90:
            status = "CRITICAL"
            risk_level = 4
            action_needed = "Immediate intervention required"
        elif temperature > 65 or server_load > 2.5 or cpu_user > 70:
            status = "WARNING"
            risk_level = 3
            action_needed = "Monitor closely and prepare for intervention"
        elif temperature > 55 or server_load > 1.5 or cpu_user > 50:
            status = "CAUTION"
            risk_level = 2
            action_needed = "Monitor for further changes"
        else:
            status = "NORMAL"
            risk_level = 1
            action_needed = "No action needed"
        
        # Generate detailed explanation
        explanation = []
        if temperature > 55:
            explanation.append(f"Temperature is elevated ({temperature}°C)")
        if server_load > 1.5:
            explanation.append(f"Server load is elevated ({server_load})")
        if cpu_user > 50:
            explanation.append(f"CPU usage is high ({cpu_user}%)")
        if not explanation:
            explanation = ["All metrics within acceptable ranges"]
        
        # Return comprehensive result
        return {
            'timestamp': timestamp,
            'status': status,
            'risk_level': risk_level,
            'action_needed': action_needed,
            'ml_insight': {
                'degradation_probability': ml_probability,
                'note': "ML model shows high sensitivity to degradation signals"
            },
            'explanation': ". ".join(explanation),
            'metrics': {
                'temperature': float(temperature),
                'server_load': float(server_load),
                'cpu_user': float(cpu_user),
                'cpu_system': float(cpu_system),
                'cpu_iowait': float(cpu_iowait),
                'memory_pct': float(memory_pct)
            }
        }