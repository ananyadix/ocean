from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model('my_model.h5')

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Ensure data is provided
        if not data or 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert input data to numpy array
        input_data = np.array(data['input'])
        
        # If single sample, reshape appropriately
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Make prediction
        predictions = model.predict(input_data)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
        
        # Prepare response
        response = []
        for i in range(len(predicted_classes)):
            response.append({
                'class_index': int(predicted_classes[i]),
                'class_label': predicted_labels[i]
            })
        
        return jsonify({'predictions': response}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host=0.0.0.0, port=1888)
