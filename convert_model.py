import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def convert_h5_to_keras():
    """Convert H5 model to Keras native format"""
    
    try:
        print("🔄 Loading H5 model...")
        
        # Load the H5 model
        model = load_model('results/models/simple_cnn.h5')
        print("✅ H5 model loaded")
        
        # Force build by running dummy prediction
        dummy_input = tf.random.normal((1, 224, 224, 3))
        _ = model(dummy_input)
        print("✅ Model built successfully")
        
        # Save in Keras native format (.keras extension)
        os.makedirs('results/models', exist_ok=True)
        model.save('results/models/simple_cnn.keras')  # ✅ Fixed: No save_format argument
        print("✅ Model saved in Keras native format")
        
        # Test loading the converted model
        loaded_model = tf.keras.models.load_model('results/models/simple_cnn.keras')
        print("✅ Conversion verified - model loads correctly")
        
        print("\n🎉 Model conversion complete!")
        print("   Use 'results/models/simple_cnn.keras' for feature extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_h5_to_keras()
