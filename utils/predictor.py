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

        # Get absolute project base directory
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Use NEW .keras model format
        self.model_path = os.path.join(
            self.BASE_DIR, "model", "model.keras"
        )

        self.class_names_path = os.path.join(
            self.BASE_DIR, "model", "class_names.json"
        )

        print("\n===== Predictor Initialization =====")
        print("BASE_DIR:", self.BASE_DIR)
        print("Model path:", self.model_path)
        print("Class names path:", self.class_names_path)
        print("Model exists:", os.path.exists(self.model_path))
        print("Class names exists:", os.path.exists(self.class_names_path))
        print("===================================\n")

        self.load_resources()


    def load_resources(self):

        # Load class names
        try:

            if os.path.exists(self.class_names_path):

                with open(self.class_names_path, "r") as f:
                    self.class_names = json.load(f)

                print(f"SUCCESS: Loaded {len(self.class_names)} classes")

            else:
                print("ERROR: class_names.json NOT FOUND")

        except Exception as e:

            print("ERROR loading class names:", e)


        # Load model (.keras format)
        try:

            if os.path.exists(self.model_path):

                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False
                )

                print("SUCCESS: Model loaded successfully")

            else:
                print("ERROR: model.keras NOT FOUND")

        except Exception as e:

            print("ERROR loading model:", e)
            self.model = None


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

            return {
                "error": "Prediction failed"
            }


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


# Global instance
predictor = Predictor()