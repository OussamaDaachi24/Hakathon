# ai_models/test_server_monitor.py
from server_monitor import ServerHealthMonitor
import os

def test_server_monitor():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "balanced_server_model")
        
        print(f"Looking for model in: {model_dir}")
        print(f"Files in directory: {os.listdir(model_dir)}")
        
        # Initialize the monitor
        print("Initializing ServerHealthMonitor...")
        monitor = ServerHealthMonitor(model_dir=model_dir)
        print("Successfully loaded model.")
        
        # Test various scenarios
        test_cases = [
            # temp, load, cpu_user, description
            (25.0, 0.1, 1.0, "Very Low Stress"),
            (35.0, 0.5, 10.0, "Low Stress"),
            (45.0, 1.0, 25.0, "Normal Operation"),
            (55.0, 1.5, 40.0, "Moderate Stress"),
            (65.0, 2.5, 60.0, "High Stress"),
            (75.0, 3.5, 80.0, "Very High Stress"),
            (85.0, 4.5, 95.0, "Critical Stress")
        ]
        
        print("\nTESTING SERVER HEALTH MONITOR")
        print("=" * 80)
        print(f"{'SCENARIO':<20} {'STATUS':<10} {'RISK':<6} {'ACTION NEEDED'}")
        print("-" * 80)
        
        for temp, load, cpu, desc in test_cases:
            result = monitor.predict(temperature=temp, server_load=load, cpu_user=cpu)
            
            print(f"{desc:<20} {result['status']:<10} {result['risk_level']:<6} {result['action_needed']}")
        
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"Error testing server monitor: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_server_monitor()