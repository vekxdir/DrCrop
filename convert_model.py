import tensorflow as tf
import os

def convert_to_saved_model():
    model_path = 'model/model.keras'
    h5_path = 'model/model.h5'
    
    # Try to load the best available local model
    try:
        if os.path.exists(model_path):
            print(f"Loading from {model_path}...")
            # We use the custom_objects fix that worked before
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'InputLayer': tf.keras.layers.InputLayer}
            )
        else:
            print(f"Loading from {h5_path}...")
            model = tf.keras.models.load_model(
                h5_path, 
                compile=False,
                custom_objects={'InputLayer': tf.keras.layers.InputLayer}
            )
        
        print("Model loaded successfully. Saving as SavedModel format...")
        save_dir = 'model/saved_model_v1'
        model.save(save_dir, save_format='tf')
        print(f"Successfully saved to {save_dir}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_to_saved_model()
