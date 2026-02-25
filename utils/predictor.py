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

        # Absolute base directory (important for Render)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Absolute paths
        self.model_path = os.path.join(BASE_DIR, "model", "model.h5")
        self.class_names_path = os.path.join(BASE_DIR, "model", "class_names.json")

        # Debug logs (will show in Render logs)
        print("=" * 50)
        print("Predictor Initialization")
        print("BASE_DIR:", BASE_DIR)
        print("Model path:", self.model_path)
        print("Class names path:", self.class_names_path)
        print("Model exists:", os.path.exists(self.model_path))
        print("Class names exists:", os.path.exists(self.class_names_path))
        print("=" * 50)

        # Load model and class names
        self.load_resources()

    def load_resources(self):
        # Load Class Names
        if os.path.exists(self.class_names_path):
            try:
                with open(self.class_names_path, "r") as f:
                    self.class_names = json.load(f)
                print(f"Loaded {len(self.class_names)} class names successfully.")
            except Exception as e:
                print("Error loading class names:", e)
        else:
            print("ERROR: class_names.json not found")

        # Load Model
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print("ERROR loading model:", e)
                self.model = None
        else:
            print("ERROR: model.h5 not found")
            self.model = None

    def predict(self, image_path):

        # Safety check
        if self.model is None:
            print("Prediction failed: model is None")
            return {"error": "Model not loaded"}

        # Preprocess image
        processed_img = preprocess_image(image_path)

        if processed_img is None:
            return {"error": "Image preprocessing failed"}

        try:
            # Make prediction
            predictions = self.model.predict(processed_img)

            confidence = float(np.max(predictions))
            predicted_index = int(np.argmax(predictions))

            # Unknown disease handling
            if confidence >= 0.75 and predicted_index < len(self.class_names):
                disease_key = self.class_names[predicted_index]
            else:
                disease_key = "Unknown Disease"

            # Get disease info
            info = DISEASE_DATABASE.get(
                disease_key,
                DISEASE_DATABASE["Unknown Disease"]
            )

            return self.format_result(info, confidence)

        except Exception as e:
            print("Prediction error:", e)
            return self.format_result(
                DISEASE_DATABASE["Unknown Disease"],
                0.0
            )

    def format_result(self, info, confidence):

        confidence_percent = round(confidence * 100, 2)

        # Confidence level classification
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


# Global instance used by Flask
predictor = Predictor()