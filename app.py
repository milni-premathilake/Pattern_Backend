from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes to allow requests from your React frontend

# Configure TensorFlow to use less memory and avoid version conflicts
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Print TensorFlow version for debugging
print(f"Using TensorFlow version: {tf.__version__}")

# Load models with version-specific handling
try:
    model_formats = [
        # Try different file extensions and paths in this order
        ('.keras', ''),           # shape_recognition_model.keras in current dir
        ('.h5', ''),              # shape_recognition_model.h5 in current dir  
        ('.keras', '../models/'),  # shape_recognition_model.keras in models dir
        ('.h5', '../models/')      # shape_recognition_model.h5 in models dir
    ]
    
    shape_model = None
    number_model = None
    
    # Try loading each format until successful
    for ext, path in model_formats:
        try:
            shape_path = f"{path}shape_recognition_model{ext}"
            number_path = f"{path}best_digit_reversal_model{ext}"
            
            # Check if files exist
            if not os.path.exists(shape_path) or not os.path.exists(number_path):
                print(f"Files not found: {shape_path} or {number_path}")
                continue
                
            print(f"Attempting to load models from: {shape_path} and {number_path}")
            
            # Custom object scope for backward compatibility
            shape_model = load_model(shape_path, compile=False)
            number_model = load_model(number_path, compile=False)
            
            print(f"Successfully loaded models with format: {ext} from path: {path}")
            break
        except Exception as model_error:
            print(f"Failed loading {ext} format from {path}: {model_error}")
    
    if shape_model is None or number_model is None:
        raise Exception("Could not load models in any format")
        
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Initialize with None to handle gracefully if models can't be loaded
    shape_model = None
    number_model = None

# Class labels for shape model
SHAPE_CLASSES = ['circle', 'square', 'triangle']

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running"""
    return jsonify({
        'status': 'ok',
        'models_loaded': {
            'shape_model': shape_model is not None,
            'number_model': number_model is not None
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint to process the image and make predictions"""
    # Check if models were loaded successfully
    if shape_model is None or number_model is None:
        return jsonify({
            'success': False,
            'error': 'Models could not be loaded. Check server logs.'
        }), 500

    # Get JSON data from request
    data = request.json
    
    if not data or 'image' not in data or not data['image'] or 'modelType' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing image data or model type'
        }), 400

    try:
        # Get model type (shape or number)
        model_type = data['modelType']
        
        # Process the base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Convert to image
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        
        # Process based on model type
        if model_type == 'shape':
            # Preprocess for shape model
            image_resized = image.resize((64, 64))
            image_array = np.array(image_resized.convert('RGB')) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            prediction = shape_model.predict(image_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            result = {
                'class': SHAPE_CLASSES[predicted_class_idx],
                'confidence': confidence,
                'all_scores': {SHAPE_CLASSES[i]: float(prediction[0][i]) for i in range(len(SHAPE_CLASSES))}
            }
            
        elif model_type == 'number':
            # Preprocess for number model
            image_resized = image.resize((28, 28))
            image_array = np.array(image_resized)
            
            # Invert the image (MNIST is white on black, but canvas is black on white)
            image_array = np.invert(image_array)
            
            # Normalize
            image_array = image_array / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = number_model.predict(image_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            result = {
                'class': str(predicted_class_idx),  # Convert digit to string
                'confidence': confidence,
                'all_scores': {str(i): float(prediction[0][i]) for i in range(10)}
            }
            
        else:
            return jsonify({
                'success': False,
                'error': f"Unknown model type: {model_type}"
            }), 400
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Azure App Service expects the app to run on the port specified by the PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)