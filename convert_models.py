"""
Model Converter Script

This script helps convert models between different TensorFlow formats.
It can be useful if you're encountering compatibility issues with your models.

Usage:
  python convert_models.py --input_model=/path/to/model.h5 --output_model=/path/to/output.keras
  
Requirements:
  - TensorFlow 2.x
"""

import argparse
import os
from tensorflow.keras.models import load_model, save_model
import tensorflow as tf

def convert_model(input_path, output_path, output_format="keras"):
    """
    Convert a TensorFlow model from one format to another.
    
    Args:
        input_path (str): Path to the input model file
        output_path (str): Path where the converted model will be saved
        output_format (str): Format to save the model in ('keras' or 'h5')
    """
    print(f"Loading model from {input_path}...")
    
    # Try loading with different options
    try:
        # First try with standard loading
        model = load_model(input_path)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Try with custom object scope
            model = load_model(input_path, compile=False)
            print("Loaded model without compilation")
        except Exception as e2:
            print(f"Loading without compilation failed: {e2}")
            raise Exception("Could not load model. Please check the format and TensorFlow version compatibility.")
    
    print("Model loaded successfully")
    print(f"Model Summary:")
    model.summary()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Saving model to {output_path}...")
    # Save model in the specified format
    if output_format.lower() == 'keras':
        save_model(model, output_path, save_format='keras')
    else:
        save_model(model, output_path, save_format='h5')
    
    print(f"Model successfully converted and saved to {output_path}")
    
    # Verify the saved model
    print("Verifying saved model...")
    try:
        loaded_model = load_model(output_path)
        print("Verification successful! The converted model can be loaded.")
    except Exception as e:
        print(f"Warning: Verification failed. The saved model might have issues: {e}")

def main():
    parser = argparse.ArgumentParser(description='Convert TensorFlow model formats.')
    parser.add_argument('--input_model', required=True, help='Path to the input model file')
    parser.add_argument('--output_model', required=True, help='Path to save the converted model')
    parser.add_argument('--output_format', default='keras', choices=['keras', 'h5'], 
                        help='Format to save the model in (default: keras)')
    
    args = parser.parse_args()
    
    print(f"TensorFlow version: {tf.__version__}")
    convert_model(args.input_model, args.output_model, args.output_format)

if __name__ == "__main__":
    main()