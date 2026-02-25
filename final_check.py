import tensorflow as tf
import os

model_path = 'model/model.keras'
class_names_path = 'model/class_names.json'

print("=== FINAL VERIFICATION ===")
try:
    if os.path.exists(model_path):
        # We load without custom_objects first to see if .keras format fixed the metadata issue
        model = tf.keras.models.load_model(model_path, compile=False)
        print("SUCCESS: model.keras loaded perfectly.")
        print(f"Input shape: {model.input_shape}")
        
        # Test a prediction with random data
        import numpy as np
        dummy_input = np.random.rand(1, 128, 128, 3).astype('float32')
        preds = model.predict(dummy_input, verbose=0)
        print(f"Prediction result shape: {preds.shape}")
        
        if preds.shape[1] == 15:
             print("SUCCESS: Model output classes match (15).")
        else:
             print(f"FAILURE: Model output classes ({preds.shape[1]}) do not match expected (15).")
             
    else:
        print("FAILURE: model.keras not found!")
except Exception as e:
    print(f"FAILURE during final check: {e}")
print("==========================")
