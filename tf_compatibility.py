"""
TensorFlow Compatibility Helper

This script helps diagnose and fix compatibility issues with TensorFlow models.
It provides information about your TensorFlow environment and attempts to load and save
models in formats compatible with your current TensorFlow version.

Usage:
  python tf_compatibility.py --diagnose
  python tf_compatibility.py --convert --input_model=model.h5 --output_model=model.keras
"""

import os
import sys
import argparse
import platform
import importlib.util
from pathlib import Path
import numpy as np

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def get_python_info():
    """Get information about the Python environment."""
    print_section("Python Environment")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Implementation: {platform.python_implementation()}")
    print(f"Python Build: {platform.python_build()}")
    print(f"System: {platform.system()} {platform.release()} {platform.machine()}")
    
    # Check for virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"Virtual Environment: Yes (path: {sys.prefix})")
    else:
        print("Virtual Environment: No")

def check_tensorflow():
    """Check TensorFlow installation and details."""
    print_section("TensorFlow Status")
    
    # Check if TensorFlow is installed
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        print("❌ TensorFlow is NOT installed")
        return False
    
    # Import TensorFlow and get version details
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow Version: {tf.__version__}")
        print(f"✓ TensorFlow Binary Compilation: {tf.sysconfig.get_build_info()}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU Available: Yes (Found {len(gpus)} GPUs)")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
        else:
            print("ℹ️ GPU Available: No - using CPU only")
        
        # Check CUDA version if available
        if hasattr(tf, 'test') and hasattr(tf.test, 'is_built_with_cuda'):
            if tf.test.is_built_with_cuda():
                print(f"✓ Built with CUDA: Yes")
                # Try to get CUDA version
                if hasattr(tf.sysconfig, 'get_build_info'):
                    build_info = tf.sysconfig.get_build_info()
                    if 'cuda_version' in build_info:
                        print(f"✓ CUDA Version: {build_info['cuda_version']}")
            else:
                print("ℹ️ Built with CUDA: No")
        
        return True
    except ImportError as e:
        print(f"❌ Error importing TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with TensorFlow: {e}")
        return False

def check_keras():
    """Check Keras installation and configuration."""
    print_section("Keras Status")
    
    try:
        # Try importing standalone Keras
        standalone_keras = importlib.util.find_spec("keras")
        tf_keras = None
        
        # Try importing TensorFlow's Keras
        try:
            import tensorflow as tf
            if hasattr(tf, 'keras'):
                tf_keras = tf.keras
        except:
            pass
        
        if standalone_keras:
            try:
                import keras
                print(f"✓ Standalone Keras Version: {keras.__version__}")
            except:
                print("❌ Standalone Keras detected but failed to import")
        else:
            print("ℹ️ Standalone Keras: Not installed")
            
        if tf_keras:
            print(f"✓ TensorFlow Keras Version: {tf_keras.__version__}")
            print(f"✓ Keras Backend: {tf_keras.backend.backend()}")
        else:
            print("ℹ️ TensorFlow Keras: Not available")
        
        # Identify potential conflicts
        if standalone_keras and tf_keras and keras.__version__ != tf_keras.__version__:
            print(f"⚠️ WARNING: Version mismatch between standalone Keras ({keras.__version__}) "
                  f"and TensorFlow Keras ({tf_keras.__version__})")
        
        return True
    except Exception as e:
        print(f"❌ Error checking Keras: {e}")
        return False

def check_model_formats():
    """Check which model formats are supported."""
    print_section("Model Format Support")
    
    try:
        import tensorflow as tf
        
        # Check .h5 format support
        print("HDF5 (.h5) Format:")
        try:
            import h5py
            print(f"  ✓ h5py Version: {h5py.__version__}")
            print("  ✓ HDF5 format is supported")
        except ImportError:
            print("  ❌ h5py not installed - HDF5 format may not be supported")
        
        # Check TensorFlow SavedModel format support
        print("\nSavedModel Format:")
        print("  ✓ SavedModel format is supported (native to TensorFlow)")
        
        # Check .keras format support (TF 2.4+)
        print("\nKeras (.keras) Format:")
        from packaging import version
        if version.parse(tf.__version__) >= version.parse("2.4.0"):
            print(f"  ✓ .keras format is supported (TensorFlow {tf.__version__} >= 2.4.0)")
        else:
            print(f"  ⚠️ .keras format may not be fully supported (TensorFlow {tf.__version__} < 2.4.0)")
        
        return True
    except Exception as e:
        print(f"❌ Error checking model formats: {e}")
        return False

def test_model_loading(model_path):
    """Test loading a model and show its properties."""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"\nAttempting to load model: {model_path}")
    
    try:
        import tensorflow as tf
        
        # Try different loading approaches
        try:
            # Standard loading
            model = tf.keras.models.load_model(model_path)
            print("✓ Successfully loaded model with standard loading")
        except Exception as e1:
            print(f"⚠️ Standard loading failed: {e1}")
            try:
                # Try without compilation
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✓ Successfully loaded model without compilation")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                return False
        
        # Show model information
        print("\nModel Information:")
        print(f"- Model Type: {type(model).__name__}")
        
        # Show input and output shapes
        if hasattr(model, 'input_shape'):
            print(f"- Input Shape: {model.input_shape}")
        if hasattr(model, 'output_shape'):
            print(f"- Output Shape: {model.output_shape}")
        
        # Show layers
        print("\nModel Architecture:")
        model.summary()
        
        return True
    except Exception as e:
        print(f"❌ Error during model loading test: {e}")
        return False

def convert_model(input_path, output_path):
    """Convert a model to be compatible with the current TensorFlow version."""
    if not os.path.exists(input_path):
        print(f"❌ Input model file not found: {input_path}")
        return False
    
    print(f"\nConverting model from {input_path} to {output_path}")
    
    try:
        import tensorflow as tf
        
        # Determine output format from extension
        output_format = Path(output_path).suffix.lower()
        
        # Load input model
        try:
            print("Loading source model...")
            model = tf.keras.models.load_model(input_path, compile=False)
        except Exception as e:
            print(f"❌ Failed to load source model: {e}")
            return False
        
        # Generate a test input for functional verification
        try:
            input_shape = model.input_shape
            if isinstance(input_shape, tuple):
                # Single input model
                test_input = np.ones(input_shape, dtype=np.float32) * 0.5
                test_output = model.predict(test_input)
            elif isinstance(input_shape, list):
                # Multiple input model
                test_inputs = [np.ones(shape, dtype=np.float32) * 0.5 for shape in input_shape]
                test_output = model.predict(test_inputs)
            else:
                test_input = None
                test_output = None
        except Exception as e:
            print(f"⚠️ Could not create test input for verification: {e}")
            test_input = None
            test_output = None
        
        # Save model in the specified format
        print(f"Saving model to {output_path}...")
        if output_format == '.keras':
            tf.keras.models.save_model(model, output_path, save_format='keras')
        elif output_format == '.h5':
            tf.keras.models.save_model(model, output_path, save_format='h5')
        else:  # Default to SavedModel format
            tf.keras.models.save_model(model, output_path)
        
        # Verify the saved model
        print("Verifying saved model...")
        try:
            converted_model = tf.keras.models.load_model(output_path, compile=False)
            print("✓ Verification successful: Model loads correctly")
            
            # Verify prediction consistency if we have test data
            if test_input is not None and test_output is not None:
                if isinstance(test_input, list):
                    converted_output = converted_model.predict(test_inputs)
                else:
                    converted_output = converted_model.predict(test_input)
                
                # Check if outputs are similar
                if isinstance(converted_output, list):
                    matches = all(np.allclose(a, b, atol=1e-5) for a, b in zip(test_output, converted_output))
                else:
                    matches = np.allclose(test_output, converted_output, atol=1e-5)
                
                if matches:
                    print("✓ Functional verification: Model outputs match before and after conversion")
                else:
                    print("⚠️ Functional verification: Model outputs differ before and after conversion")
            
            return True
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error during model conversion: {e}")
        return False

def run_diagnose():
    """Run full diagnosis of TensorFlow environment."""
    get_python_info()
    tf_available = check_tensorflow()
    if tf_available:
        check_keras()
        check_model_formats()

def main():
    parser = argparse.ArgumentParser(description='TensorFlow Compatibility Helper')
    parser.add_argument('--diagnose', action='store_true', help='Run full diagnosis')
    parser.add_argument('--test_model', help='Test loading a specific model')
    parser.add_argument('--convert', action='store_true', help='Convert a model to be compatible with current TensorFlow')
    parser.add_argument('--input_model', help='Input model path for conversion')
    parser.add_argument('--output_model', help='Output model path for conversion')
    
    args = parser.parse_args()
    
    if args.diagnose:
        run_diagnose()
    elif args.test_model:
        check_tensorflow()
        test_model_loading(args.test_model)
    elif args.convert:
        if not args.input_model or not args.output_model:
            print("❌ Error: --input_model and --output_model are required for conversion")
            return
        check_tensorflow()
        convert_model(args.input_model, args.output_model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()