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


        # Load model by rebuilding architecture (The bulletproof fix)
        try:
            print("--- Rebuilding Model Architecture ---", flush=True)
            self.model = self.build_model_architecture()
            
            # Try to load weights from .h5 (most compatible)
            h5_path = self.model_path.replace(".keras", ".h5")
            target_weights = h5_path if os.path.exists(h5_path) else self.model_path
            
            print(f"Loading weights from: {target_weights}", flush=True)
            self.model.load_weights(target_weights)
            print("MODEL LOADED SUCCESSFULLY (BY ARCHITECTURE REBUILD)", flush=True)

        except Exception as e:
            print(f"CRITICAL MODEL LOAD ERROR: {e}", flush=True)
            self.model = None

    def build_model_architecture(self):
        """Hard-coded MobileNetV2 architecture matching train.py exactly."""
        from tensorflow.keras import layers, applications

        IMG_SIZE = (128, 128)
        TARGET_IMG_SIZE = (224, 224)
        NUM_CLASSES = 15

        # 1. Input layer
        inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

        # 2. Sequential Data Augmentation (topology matching even if pass-through)
        # Random layers in inference mode act as identity/pass-through
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)

        # 3. Preprocessing
        x = layers.Resizing(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1])(x)
        x = layers.Rescaling(1./127.5, offset=-1)(x)

        # 4. Base Model (weights=None to avoid downloads, topology must match)
        base_model = applications.MobileNetV2(
            input_shape=TARGET_IMG_SIZE + (3,),
            include_top=False,
            weights=None
        )
        x = base_model(x, training=False)

        # 5. Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)


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