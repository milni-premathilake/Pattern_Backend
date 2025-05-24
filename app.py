from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import os
import tempfile
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize variables
shape_model = None
number_model = None
model_load_error = None

def load_models():
    global shape_model, number_model, model_load_error
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Configure TensorFlow for Azure
        tf.get_logger().setLevel('ERROR')  # Reduce log verbosity
        
        # Try Azure Storage first
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if connection_string:
            print("Attempting to load models from Azure Storage...")
            from azure.storage.blob import BlobServiceClient
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_name = "models"
            
            # Download and load models
            model_files = [
                ('shape_recognition_model.keras', 'shape'),
                ('shape_recognition_model.h5', 'shape'),
                ('best_digit_reversal_model.keras', 'number'),
                ('best_digit_reversal_model.h5', 'number')
            ]
            
            for filename, model_type in model_files:
                try:
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name, blob=filename
                    )
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(filename)[1]
                    )
                    
                    # Download blob
                    with open(temp_file.name, "wb") as download_file:
                        download_file.write(blob_client.download_blob().readall())
                    
                    # Load model
                    model = tf.keras.models.load_model(temp_file.name, compile=False)
                    
                    if model_type == 'shape' and shape_model is None:
                        shape_model = model
                        print(f"Shape model loaded from {filename}")
                    elif model_type == 'number' and number_model is None:
                        number_model = model
                        print(f"Number model loaded from {filename}")
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)
                    
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
        
        # Fallback to local files
        if shape_model is None or number_model is None:
            print("Attempting to load models from local files...")
            local_files = [
                ('shape_recognition_model.keras', 'shape'),
                ('shape_recognition_model.h5', 'shape'),
                ('best_digit_reversal_model.keras', 'number'),
                ('best_digit_reversal_model.h5', 'number')
            ]
            
            for filename, model_type in local_files:
                if os.path.exists(filename):
                    try:
                        model = tf.keras.models.load_model(filename, compile=False)
                        if model_type == 'shape' and shape_model is None:
                            shape_model = model
                            print(f"Shape model loaded from local {filename}")
                        elif model_type == 'number' and number_model is None:
                            number_model = model
                            print(f"Number model loaded from local {filename}")
                    except Exception as e:
                        print(f"Failed to load local {filename}: {e}")
        
        if shape_model is None or number_model is None:
            model_load_error = "Could not load all required models"
            print(model_load_error)
        else:
            print("All models loaded successfully")
            
    except Exception as e:
        model_load_error = f"Model loading error: {str(e)}"
        print(model_load_error)

# Load models on startup
load_models()

SHAPE_CLASSES = ['circle', 'square', 'triangle']

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'models_loaded': {
            'shape_model': shape_model is not None,
            'number_model': number_model is not None
        },
        'model_load_error': model_load_error,
        'environment': os.environ.get('WEBSITE_SITE_NAME', 'local')
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if shape_model is None or number_model is None:
        return jsonify({
            'success': False,
            'error': f'Models not available. Error: {model_load_error}'
        }), 500

    data = request.json
    if not data or 'image' not in data or 'modelType' not in data:
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
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)