"""
Test Models Script

This script helps test if your models can be loaded correctly with your current
TensorFlow installation. It also prints model information and attempts to run 
a simple inference task to verify functionality.

Usage:
  python test_models.py --model_path=/path/to/model.keras --model_type=shape
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import time
import os

def test_shape_model(model_path):
    """Test a shape detection model"""
    print(f"Testing shape model: {model_path}")
    
    try:
        # Try loading the model
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Create a sample test image (64x64 blank image)
        print("\nCreating test input...")
        test_input = np.ones((1, 64, 64, 3), dtype=np.float32) * 0.5  # Gray image
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        prediction = model.predict(test_input)
        end_time = time.time()
        
        # Process results
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        predicted_class = np.argmax(prediction[0])
        class_names = ['ellipse', 'rectangle', 'triangle']
        
        print(f"\nInference completed in {inference_time:.2f} ms")
        print(f"Predicted class: {class_names[predicted_class]} (class index: {predicted_class})")
        print(f"Confidence scores: {prediction[0]}")
        
        return True
    
    except Exception as e:
        print(f"Error testing shape model: {e}")
        return False

def test_number_model(model_path):
    """Test a number detection model"""
    print(f"Testing number model: {model_path}")
    
    try:
        # Try loading the model
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Create a sample test image (28x28 blank image)
        print("\nCreating test input...")
        test_input = np.ones((1, 28, 28, 1), dtype=np.float32) * 0.5  # Gray image
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        prediction = model.predict(test_input)
        end_time = time.time()
        
        # Process results
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        predicted_class = np.argmax(prediction[0])
        
        print(f"\nInference completed in {inference_time:.2f} ms")
        print(f"Predicted digit: {predicted_class}")
        print(f"Confidence scores: {prediction[0]}")
        
        return True
    
    except Exception as e:
        print(f"Error testing number model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test TensorFlow models.')
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--model_type', required=True, choices=['shape', 'number'], 
                        help='Type of model to test')
    
    args = parser.parse_args()
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Testing model: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return
    
    if args.model_type == 'shape':
        success = test_shape_model(args.model_path)
    else:
        success = test_number_model(args.model_path)
    
    if success:
        print("\n✅ Model test completed successfully!")
    else:
        print("\n❌ Model test failed.")

if __name__ == "__main__":
    main()