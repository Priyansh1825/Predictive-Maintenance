from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Load model and preprocessing objects
try:
    model = joblib.load('models/predictive_maintenance_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_selector = joblib.load('models/feature_selector.pkl')
    
    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    print("✅ Models loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    print(f"🔧 Features: {len(feature_columns)}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    feature_selector = None
    feature_columns = []

@app.route('/')
def home():
    """Home page with information about the API"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions from JSON data"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}',
                'required_features': feature_columns
            }), 400
        
        # Select and order features correctly
        input_data = input_data[feature_columns]
        
        # Preprocess data
        scaled_data = scaler.transform(input_data)
        selected_features = feature_selector.transform(scaled_data)
        
        # Make prediction
        prediction = model.predict(selected_features)[0]
        probability = model.predict_proba(selected_features)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability_failure': float(probability[1]),
            'probability_no_failure': float(probability[0]),
            'prediction_label': 'FAILURE' if prediction == 1 else 'NO FAILURE',
            'confidence': float(max(probability))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Make predictions from form data"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert to proper data types
        input_data = {}
        for key, value in form_data.items():
            try:
                input_data[key] = float(value)
            except ValueError:
                input_data[key] = value
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            return f"Error: Missing features: {list(missing_features)}", 400
        
        # Select and order features correctly
        input_df = input_df[feature_columns]
        
        # Preprocess data
        scaled_data = scaler.transform(input_df)
        selected_features = feature_selector.transform(scaled_data)
        
        # Make prediction
        prediction = model.predict(selected_features)[0]
        probability = model.predict_proba(selected_features)[0]
        
        # Prepare result
        result = {
            'prediction': 'FAILURE' if prediction == 1 else 'NO FAILURE',
            'failure_probability': f"{probability[1]:.2%}",
            'no_failure_probability': f"{probability[0]:.2%}",
            'confidence': f"{max(probability):.2%}"
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': len(feature_columns) > 0
    })

@app.route('/features')
def features():
    """Return required features for prediction"""
    return jsonify({
        'required_features': feature_columns,
        'feature_count': len(feature_columns)
    })

def create_templates():
    """Create HTML templates if they don't exist"""
    
    # Create index.html
    index_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Predictive Maintenance API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .form-group { margin: 10px 0; }
        label { display: inline-block; width: 200px; }
        input { padding: 5px; width: 200px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .result { background: #e8f5e8; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Predictive Maintenance System</h1>
        
        <div class="card">
            <h2>Machine Failure Prediction</h2>
            <form action="/predict_form" method="post">
                <div class="form-group">
                    <label>Air Temperature [K]:</label>
                    <input type="number" name="Air temperature [K]" step="0.1" value="298.1" required>
                </div>
                <div class="form-group">
                    <label>Process Temperature [K]:</label>
                    <input type="number" name="Process temperature [K]" step="0.1" value="308.6" required>
                </div>
                <div class="form-group">
                    <label>Rotational Speed [rpm]:</label>
                    <input type="number" name="Rotational speed [rpm]" value="1551" required>
                </div>
                <div class="form-group">
                    <label>Torque [Nm]:</label>
                    <input type="number" name="Torque [Nm]" step="0.1" value="42.8" required>
                </div>
                <div class="form-group">
                    <label>Tool Wear [min]:</label>
                    <input type="number" name="Tool wear [min]" value="0" required>
                </div>
                <div class="form-group">
                    <label>Type (0=L, 1=M, 2=H):</label>
                    <input type="number" name="Type_encoded" min="0" max="2" value="0" required>
                </div>
                <button type="submit">Predict Failure</button>
            </form>
        </div>

        <div class="card">
            <h2>API Usage</h2>
            <p><strong>JSON Endpoint:</strong> POST /predict</p>
            <p><strong>Health Check:</strong> GET /health</p>
            <p><strong>Features:</strong> GET /features</p>
        </div>
    </div>
</body>
</html>
    '''
    
    # Create result.html
    result_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .result { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .failure { background: #f8d7da; }
        .back-btn { background: #6c757d; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Prediction Result</h1>
        
        <div class="result {% if result.prediction == 'FAILURE' %}failure{% endif %}">
            <h2>Prediction: {{ result.prediction }}</h2>
            <p><strong>Failure Probability:</strong> {{ result.failure_probability }}</p>
            <p><strong>No Failure Probability:</strong> {{ result.no_failure_probability }}</p>
            <p><strong>Confidence:</strong> {{ result.confidence }}</p>
        </div>
        
        <a href="/" class="back-btn">Make Another Prediction</a>
    </div>
</body>
</html>
    '''
    
    # Write template files
    with open('templates/index.html', 'w') as f:
        f.write(index_html)
    
    with open('templates/result.html', 'w') as f:
        f.write(result_html)
    
    print("✅ HTML templates created!")

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create simple HTML templates
    create_templates()
    
    print("🚀 Starting Predictive Maintenance API...")
    print("📊 Endpoints:")
    print("   - GET  / : Home page")
    print("   - POST /predict : JSON prediction API")
    print("   - POST /predict_form : Form prediction")
    print("   - GET  /health : Health check")
    print("   - GET  /features : Required features")
    
    app.run(debug=True, host='0.0.0.0', port=5000)