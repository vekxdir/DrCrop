import os
import json
import numpy as np
import tensorflow as tf
from utils.disease_info import DISEASE_DATABASE
from utils.preprocess import preprocess_image


class Predictor:

    def __init__(self):

        self.model = None
        self.class_names = []

        # absolute base directory
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # USE model.keras
        self.model_path = os.path.join(
            self.BASE_DIR,
            "model",
            "model.keras"
        )

        self.class_names_path = os.path.join(
            self.BASE_DIR,
            "model",
            "class_names.json"
        )

        print("\n=== Predictor Init ===")
        print("Model path:", self.model_path)
        print("Model exists:", os.path.exists(self.model_path))
        print("======================\n")

        self.load_resources()


    def load_resources(self):

        # Load class names
        try:

            if os.path.exists(self.class_names_path):

                with open(self.class_names_path, "r") as f:
                    self.class_names = json.load(f)

                print("Class names loaded")

            else:

                print("class_names.json NOT FOUND")

        except Exception as e:

            print("Class names load error:", e)


        # Load model with multiple fallbacks for cross-platform compatibility
        try:
            if os.path.exists(self.model_path):
                print(f"Attempting to load model from: {self.model_path}")
                self.model = self.robust_load(self.model_path)
                
                if self.model:
                    print("MODEL LOADED SUCCESSFULLY")
                else:
                    # Try .h5 fallback if .keras failed
                    h5_path = self.model_path.replace(".keras", ".h5")
                    if os.path.exists(h5_path):
                        print(f"Attempting fallback to: {h5_path}")
                        self.model = self.robust_load(h5_path)
                        if self.model:
                            print("H5 MODEL LOADED SUCCESSFULLY")

            if not self.model:
                print("Model could not be loaded after all attempts.")
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    print(f"Contents of {model_dir}: {os.listdir(model_dir)}")

        except Exception as e:
            print(f"CRITICAL MODEL LOAD ERROR: {e}")
            self.model = None

    def robust_load(self, path):
        """Tries various methods to load the model, including deep config patching for Keras 2/3 compatibility."""
        import h5py
        import json
        import re

        print(f"--- robust_load starting for {path} ---", flush=True)

        # 1. Try standard load first (Bypass InputLayer issue if registry is fixed)
        try:
            return tf.keras.models.load_model(
                path, 
                compile=False,
                custom_objects={'InputLayer': tf.keras.layers.InputLayer}
            )
        except Exception as e:
            print(f"Standard load failed for {path}: {e}", flush=True)

        # 2. Deep Manual Config Patching
        try:
            print("Attempting Deep Configuration Patching...", flush=True)
            # Patching is most reliable with .h5 files
            work_path = path
            if not path.endswith('.h5'):
                h5_compat = path.replace('.keras', '.h5')
                if os.path.exists(h5_compat):
                    work_path = h5_compat
                else:
                    print("No .h5 file found for deep patching fallback.", flush=True)

            with h5py.File(work_path, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    print("No model_config found in H5 attributes.", flush=True)
                    return None
                
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                print("Original config size:", len(model_config), flush=True)

                # --- THE DEEP CLEANSE ---
                
                # A. Keras 3 uses 'batch_shape', Keras 2 expects 'batch_input_shape'
                model_config = model_config.replace('"batch_shape"', '"batch_input_shape"')
                
                # B. Keras 3 module paths (keras.src) don't exist in Keras 2
                model_config = model_config.replace('keras.src.models.functional', 'keras.models')
                model_config = model_config.replace('keras.src.layers', 'keras.layers')
                
                # C. Keras 3 uses 'Functional', Keras 2 uses 'Model' or 'Sequential'
                # Note: Functional -> Model is the most common mismatch
                model_config = model_config.replace('"class_name": "Functional"', '"class_name": "Model"')

                # D. STRIP DTypePolicy: Keras 3 saves dtype as a complex object, Keras 2 wants a string
                # Example: "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, ...}
                # We replace the whole dict with just "float32"
                pattern = r'\{\s*"module":\s*"keras",\s*"class_name":\s*"DTypePolicy",\s*"config":\s*\{\s*"name":\s*"([^"]+)"\s*\},\s*"registered_name":\s*null\s*\}'
                model_config = re.sub(pattern, r'"\1"', model_config)
                
                # Fallback for slightly different DTypePolicy formats
                model_config = re.sub(r'\{\s*"class_name":\s*"DTypePolicy",\s*"config":\s*\{\s*"name":\s*"([^"]+)"\s*\}.*?\}', r'"\1"', model_config)

                print("Patched config size:", len(model_config), flush=True)

                # Attempt to recreate from patched JSON
                try:
                    # We pass InputLayer to custom_objects just in case
                    model = tf.keras.models.model_from_json(
                        model_config, 
                        custom_objects={'InputLayer': tf.keras.layers.InputLayer}
                    )
                    model.load_weights(work_path)
                    print("DEEP PATCH SUCCESS: Model reconstructed and weights loaded.", flush=True)
                    return model
                except Exception as e_json:
                    print(f"Deep patch Reconstruction failed: {e_json}", flush=True)
                    return None
                    
        except Exception as patch_e:
            print(f"Deep patching process failed: {patch_e}", flush=True)
            return None


    def predict(self, image_path):

        if self.model is None:
            return {"error": "Model not loaded"}

        processed_img = preprocess_image(image_path)

        if processed_img is None:
            return {"error": "Image preprocessing failed"}

        try:

            predictions = self.model.predict(processed_img)

            confidence = float(np.max(predictions))

            predicted_index = int(np.argmax(predictions))

            if confidence >= 0.75 and predicted_index < len(self.class_names):

                disease_key = self.class_names[predicted_index]

            else:

                disease_key = "Unknown Disease"


            info = DISEASE_DATABASE.get(
                disease_key,
                DISEASE_DATABASE["Unknown Disease"]
            )

            return self.format_result(info, confidence)

        except Exception as e:

            print("Prediction error:", e)

            return {"error": "Prediction failed"}


    def format_result(self, info, confidence):

        confidence_percent = round(confidence * 100, 2)

        if confidence_percent >= 90:

            confidence_level = "High Confidence"
            confidence_class = "success"

        elif confidence_percent >= 75:

            confidence_level = "Moderate Confidence"
            confidence_class = "warning"

        else:

            confidence_level = "Low Confidence"
            confidence_class = "danger"

        return {

            "disease_name": info["name"],
            "crop": info["crop"],
            "risk_level": info["risk"],

            "confidence": f"{confidence_percent}%",
            "confidence_score": confidence_percent,

            "confidence_level": confidence_level,
            "confidence_class": confidence_class,

            "description": info["description"],
            "causes": info["causes"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        }


# global instance
predictor = Predictor()