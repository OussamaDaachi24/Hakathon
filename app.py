from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import datetime
import json
import time
import ai_models.temp_predictor as temp
import ai_models.face_recognition  as recognize

app = Flask(__name__)

# Database path
DB_PATH = 'instance/db.sqlite'
os.makedirs('instance', exist_ok=True)

# Temperature threshold (in Celsius)
TEMP_THRESHOLD = 28.0  # not defined yet

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS temperature_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NULL,
            server_load REAL , 
            fans_speed REAL             
        );
        
        
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person_id TEXT NOT NULL,  
            action TEXT NOT NULL
        );
        

        CREATE TABLE IF NOT EXISTS authorized_personnel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            rfid_id TEXT,
            face_id TEXT
        );
        
    ''')
    
    # Add some sample authorized personnel
    conn.commit()
    conn.close()

# Web Interface Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ESP32 Temperature Sensor API
@app.route('/api/temperature', methods=['POST']) # record the temperature sent by ESP-32
def record_temperature():
    data = request.json
    
    if not data or 'temperature' not in data :
        return jsonify({'error': 'Missing required data'}), 400
    
    time = datetime.datetime.now().isoformat()
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    server_load = data.get('server_load', 0.0)
    
    # Store temperature data in database
    conn = get_db()
    conn.execute(
        'INSERT INTO temperature_logs ("timestamp",temperature, humidity, server_load) VALUES (?, ?, ?, ?)',
        (time, temperature, humidity, server_load)
    )
    conn.commit()
    
    # Get historical temperature data for AI prediction
    # get the historical data for the prediction model
    historical_data = conn.execute(
    'SELECT "timestamp", temperature, humidity, server_load FROM temperature_logs ' +
    'ORDER BY "timestamp" DESC LIMIT 24'
).fetchall()
    
    # Format data for AI model
    temp_history = [dict(row) for row in historical_data]
    
    # Check if we have enough historical data
    response = {'message': 'Temperature recorded', '"timestamp"': time}
    
    if len(temp_history) >= 5:  # Need enough data for prediction
        # Call the imported AI model to predict temperature and fan speed
        prediction = temp.predict_temperature(temp_history)
        if prediction and prediction['temperature']  and prediction['fan_speed'] :
            # Extract predicted values from the model result
            predicted_temp = prediction['temperature']
            fan_speed = prediction['fan_speed']
            
            # Record fan state based on AI recommendation
            conn.execute(
                'INSERT INTO temperature_logs (fan_speed) VALUES (?)',
                (fan_speed,)  # Fixed: Added comma to make it a tuple
            )
            
            # Add prediction info to response
            response['action'] = 'ai_controlled_cooling'
            response['fan_speed'] = fan_speed
            response['prediction'] = prediction
        else:
            # Fallback for when prediction fails
            if temperature > TEMP_THRESHOLD:
                fan_speed = min(100, int((temperature - TEMP_THRESHOLD) * 10 + 50))
            else:
                fan_speed = 20
                
            response['action'] = 'fallback_cooling'
            response['fan_speed'] = fan_speed
            response['note'] = 'Using fallback temperature control (prediction failed)'
    else:
        # Fallback for when not enough data exists
        if temperature > TEMP_THRESHOLD:
            fan_speed = min(100, int((temperature - TEMP_THRESHOLD) * 10 + 50))
        else:
            fan_speed = 20
            
        # Record fan state
        conn.execute(
            'INSERT INTO temperature_logs (fan_speed) VALUES (?)',
            ( fan_speed)
        )
        conn.commit()
        
        response['action'] = 'fallback_cooling' # will tell the fans to start
        response['fan_speed'] = fan_speed  # at this speed
        response['note'] = 'Using fallback temperature control (insufficient data for AI)'
    
    conn.close()
    return jsonify(response)

@app.route('/api/temperature', methods=['GET'])
def get_temperature_history():
    limit = request.args.get('limit', default=100, type=int)
    
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM temperature_logs ORDER BY timestamp DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    
    return jsonify([dict(row) for row in rows])



@app.route('/api/temperature/latest', methods=['GET'])
def get_latest_temperature():
    conn = get_db()
    row = conn.execute(
        'SELECT * FROM temperature_logs ORDER BY timestamp DESC LIMIT 1'
    ).fetchone()
    conn.close()
    
    if row:
        return jsonify(dict(row))
    else:
        return jsonify({'error': 'No temperature data available'}), 404




@app.route('/api/temperature/predict', methods=['GET'])
def predict_temperature_route():
    conn = get_db()
    # Get historical temperature data
    rows = conn.execute(
        'SELECT timestamp, temperature, humidity, server_load FROM temperature_logs ' +
        'ORDER BY timestamp DESC LIMIT 24'
    ).fetchall()
    conn.close()
    
    if len(rows) < 5:  # Need enough data for prediction
        return jsonify({'error': 'Insufficient data for prediction'}), 400
    
    # Format data for AI model
    historical_data = [dict(row) for row in rows]
    
    # Get prediction from AI model
    prediction = temp.predict_temperature(historical_data)
    
    return jsonify({
        'current_temperature': historical_data[0]['temperature'],
        'predicted_temperature': prediction['temperature'],
        'predicted_timestamp': prediction['timestamp'],
        'confidence': prediction.get('confidence', 0.8)
    })



# ESP32 Security API (RFID and Camera)
@app.route('/api/security/rfid', methods=['POST'])  # ESP-32 sends rfid to the backend
def process_rfid():
    data = request.json
    
    if not data or 'rfid_id' not in data:
        return jsonify({'error': 'Missing RFID data'}), 400
    
    rfid_id = data['rfid_id']
    timestamp = datetime.datetime.now().isoformat()
    
    # Check if RFID is authorized
    conn = get_db()
    person = conn.execute(
        'SELECT * FROM authorized_personnel WHERE rfid_id = ?',
        (rfid_id,)
    ).fetchone()
    
    if person:  # send back to the ESP-32 to open the camera
        
        # ADDED: Trigger camera for facial recognition
        return jsonify({
            'access': 'pending',  # Changed to pending until face verification
            'person_id': person['person_id'],
            'name': person['name'],
            'timestamp': timestamp,
            'action': 'activate_camera'  # ADDED: Signal ESP32 to activate camera
        })
    else:
        # Unauthorized access attempt
        conn.execute(
            'INSERT INTO access_logs (timestamp, person_id, action) VALUES (?, ?, ?)',
            (timestamp, 'unknown', 'denied')
        )
        conn.commit()
        
        return jsonify({
            'access': 'denied',  # will be sent to the ESP-32
            'timestamp': timestamp
        })


@app.route('/api/security/face', methods=['POST'])
def process_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    timestamp = datetime.datetime.now().isoformat()
    
    # Save image temporarily
    temp_image_path = f"temp_face_{int(time.time())}.jpg"
    image.save(temp_image_path)
    
    try:
        # Process with face recognition AI
        recognition_result = recognize.recognize_face(temp_image_path)
        
        conn = get_db()
        
        if recognition_result['recognized']:
            face_id = recognition_result['person_id']
            confidence = recognition_result['confidence']
            
            # Check if face is authorized
            person = conn.execute(
                'SELECT * FROM authorized_personnel WHERE face_id = ?',
                (face_id,)
            ).fetchone()
            
            if person:
                # Authorized person
                conn.execute(
                    'INSERT INTO access_logs (timestamp, person_id, action) VALUES (?, ?, ?)',
                    (timestamp, person['person_id'],'recognized')
                )
                
                result = {
                    'recognized': True,
                    'authorized': True,
                    'person_id': person['person_id'],
                    'name': person['name'],
                    'confidence': confidence,
                    'timestamp': timestamp
                }
            else:
                # Recognized but not authorized
                conn.execute(
                    'INSERT INTO access_logs (timestamp, person_id, access_type, action) VALUES (?, ?, ?, ?)',
                    (timestamp, 'unknown', 'face', 'unauthorized')
                )
                
                # Create security alert
                conn.execute(
                    'INSERT INTO alerts (timestamp, alert_type, description) VALUES (?, ?, ?)',
                    (timestamp, 'security', f"Unauthorized person detected in server room")
                )
                conn.commit()
                
                result = {
                    'recognized': True,
                    'authorized': False,
                    'confidence': confidence,
                    'timestamp': timestamp
                }
        else:
            # Face not recognized
            
            # TO implement : create an alert

            result = {
                'recognized': False,
                'authorized': False,
                'timestamp': timestamp
            }
        
        conn.close()
        return jsonify(result)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# Dashboard and Alert API
@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    conn = get_db()
    
    # Get latest temperature reading with fan speed
    latest_temp = conn.execute(
        'SELECT * FROM temperature_logs ORDER BY timestamp DESC LIMIT 1'
    ).fetchone()
    
    
    conn.close()
    
    return jsonify({
        'current_temperature': dict(latest_temp) if latest_temp else None,
    })


@app.route('/api/access_logs', methods=['GET'])
def get_access_logs():
    limit = request.args.get('limit', default=50, type=int)
    
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    
    return jsonify([dict(row) for row in rows])

@app.route('/api/fan/control', methods=['POST'])
def update_fan_state():
    # Get data from request
    data = request.get_json()
    speed = data.get('speed', 0)
    
    # ESP32 endpoint
    esp32_url = 'http://esp32-ip-address/fan'  # Replace with actual ESP32 IP
    
    try:
        # Send data to ESP32
        response = request.post(esp32_url, json={
            'speed': speed
        })
        
        # Return response to frontend
        return jsonify({
            'success': True,
            'fan_speed': speed
        })
        
    except Exception as e:
        # Handle errors
        return jsonify({'error': 'Failed to communicate with ESP32'}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    # Get unresolved alerts or recent alerts
    unresolved_only = request.args.get('unresolved', 'false').lower() == 'true'
    limit = request.args.get('limit', default=50, type=int)
    
    conn = get_db()
    if unresolved_only:
        rows = conn.execute(
            'SELECT * FROM alerts WHERE resolved = 0 ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        ).fetchall()
    
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.cli.command("init-db")
def init_db_command():
    """Create tables and seed data"""
    init_db()
    # Add sample data if needed
    print("Database initialized!")

# Run with: flask init-db

if __name__ == '__main__':

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)