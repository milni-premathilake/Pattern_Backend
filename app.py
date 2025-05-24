from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from azure.storage.blob import BlobServiceClient
import tempfile

app = Flask(__name__)
CORS(app)

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "models"

# Configure TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"Using TensorFlow version: {tf.__version__}")

def download_model_from_blob(blob_name):
    """Download model from Azure Blob Storage to temporary file"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob_name)[1])
        
        # Download blob to temporary file
        with open(temp_file.name, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        return temp_file.name
    except Exception as e:
        print(f"Error downloading {blob_name}: {e}")
        return None

# Load models from Azure Blob Storage
try:
    shape_model = None
    number_model = None
    
    if AZURE_STORAGE_CONNECTION_STRING:
        # Try different model file names
        model_files = [
            ('shape_recognition_model.keras', 'shape'),
            ('shape_recognition_model.h5', 'shape'),
            ('best_digit_reversal_model.keras', 'number'),
            ('best_digit_reversal_model.h5', 'number')
        ]
        
        shape_model_path = None
        number_model_path = None
        
        for filename, model_type in model_files:
            temp_path = download_model_from_blob(filename)
            if temp_path:
                if model_type == 'shape' and shape_model is None:
                    try:
                        shape_model = load_model(temp_path, compile=False)
                        shape_model_path = temp_path
                        print(f"Loaded shape model from {filename}")
                    except Exception as e:
                        print(f"Failed to load shape model from {filename}: {e}")
                        os.unlink(temp_path)
                elif model_type == 'number' and number_model is None:
                    try:
                        number_model = load_model(temp_path, compile=False)
                        number_model_path = temp_path
                        print(f"Loaded number model from {filename}")
                    except Exception as e:
                        print(f"Failed to load number model from {filename}: {e}")
                        os.unlink(temp_path)
    else:
        print("Azure Storage connection string not found in environment variables")
        
    if shape_model is None or number_model is None:
        print("Warning: Some models could not be loaded")
        
except Exception as e:
    print(f"Error loading models: {e}")
    shape_model = None
    number_model = None

# Rest of your existing code remains the same...
SHAPE_CLASSES = ['circle', 'square', 'triangle']

@app.route('/api/predict', methods=['POST'])
def predict():
    # Your existing predict function code...
    if shape_model is None or number_model is None:
        return jsonify({
            'success': False,
            'error': 'Models could not be loaded. Check server logs.'
        }), 500

    data = request.json
    
    if not data or 'image' not in data or not data['image'] or 'modelType' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing image data or model type'
        }), 400

    try:
        model_type = data['modelType']
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        if model_type == 'shape':
            image_resized = image.resize((64, 64))
            image_array = np.array(image_resized.convert('RGB')) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            prediction = shape_model.predict(image_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            result = {
                'class': SHAPE_CLASSES[predicted_class_idx],
                'confidence': confidence,
                'all_scores': {SHAPE_CLASSES[i]: float(prediction[0][i]) for i in range(len(SHAPE_CLASSES))}
            }
            
        elif model_type == 'number':
            image_resized = image.resize((28, 28))
            image_array = np.array(image_resized)
            image_array = np.invert(image_array)
            image_array = image_array / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)
            
            prediction = number_model.predict(image_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            result = {
                'class': str(predicted_class_idx),
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)