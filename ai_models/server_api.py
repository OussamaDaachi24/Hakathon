# ai_models/server_api.py
from flask import Flask, request, jsonify
from server_monitor import ServerHealthMonitor
import os

app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "balanced_server_model")

# Initialize the monitor
monitor = ServerHealthMonitor(model_dir=model_dir)

@app.route('/api/server-health', methods=['POST'])
def check_server_health():
    try:
        data = request.get_json()
        
        # Extract required parameters
        temperature = float(data.get('temperature', 0))
        server_load = float(data.get('server_load', 0))
        cpu_user = float(data.get('cpu_user', 0))
        
        # Optional parameters
        cpu_system = float(data.get('cpu_system', 0.5))
        cpu_iowait = float(data.get('cpu_iowait', 0.1))
        memory_pct = float(data.get('memory_pct', 0.2))
        
        # Get prediction
        result = monitor.predict(
            temperature=temperature,
            server_load=server_load,
            cpu_user=cpu_user,
            cpu_system=cpu_system,
            cpu_iowait=cpu_iowait,
            memory_pct=memory_pct
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'ERROR'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)