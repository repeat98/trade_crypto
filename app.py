from flask import Flask, send_from_directory, jsonify
import json
import os

app = Flask(__name__, static_folder='.')

# Load dashboard data
try:
    with open('dashboard.json', 'r') as f:
        dashboard_data = json.load(f)
except FileNotFoundError:
    print("Warning: dashboard.json not found. Using empty data.")
    dashboard_data = {}

@app.route('/')
def index():
    # serve your index.html
    return send_from_directory('.', 'index.html')

@app.route('/data')
def data():
    try:
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # listen on all interfaces on port 8000
    app.run(host='0.0.0.0', port=8000, debug=True)