import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import ssl

# Fix SSL context for downloading weights (on some environments)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configuration
DATASET_DIR = 'dataset'
MODEL_DIR = 'model'
IMG_SIZE = (128, 128)  # Matches app.py input
TARGET_IMG_SIZE = (224, 224) # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 25

def train_model():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please create a 'dataset' folder with subfolders for each class.")
        return

    print("Loading dataset...")
    # Load dataset
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
         print(f"Failed to load dataset: {e}")
         return

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    # Base Model (MobileNetV2)
    # Using include_top=False to remove the classification head
    # Using weights='imagenet' for transfer learning
    # Input shape is (224, 224, 3) for MobileNetV2
    base_model = applications.MobileNetV2(input_shape=TARGET_IMG_SIZE + (3,),
                                          include_top=False,
                                          weights='imagenet')
    
    # Freeze the base model initially
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    # Preprocessing pipeline
    # 1. Augment data
    x = data_augmentation(inputs)
    # 2. Resize to 224x224 for MobileNetV2
    x = layers.Resizing(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1])(x)
    # 3. Rescale for MobileNetV2 (-1 to 1)
    # Note: image_dataset loads 0-255. MobileNetV2 expects -1 to 1.
    x = layers.Rescaling(1./127.5, offset=-1)(x) # 0-255 -> -1 to 1
    
    # Through base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)  # Regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # Compile
    base_learning_rate = 0.0001 # Lower learning rate for fine-tuning/transfer learning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    
    # Use callbacks for saving best model during training if desired, but user just asked to save at end or 'model/model.h5'.
    # We will stick to the basic requirement first.

    # Fine-tuning (Optional but recommended for 90%+ accuracy)
    # Unfreeze the base_model and train again with a very low learning rate
    # For now, let's stick to the 25 epochs on the frozen base + head, 
    # unless accuracy is low. With 25 epochs on a good dataset, head training should be decent.
    # To hit 90%+, fine-tuning is usually key. Let's add a small fine-tuning step?
    # User said "Freeze base layers initially", implying unfreezing later? 
    # Or just "initially" means "start with frozen".
    # I will add a fine-tuning block if accuracy is not satisfying, or just include it as part of the script.
    # Let's add a second phase of training (Fine-tuning) automatically.
    
    print("\nInitial training complete. Starting Fine-Tuning phase...")
    
    base_model.trainable = True
    # Freeze the earlier layers, only train the last few
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy'])
                  
    total_epochs = EPOCHS + 10 # Add 10 more epochs for fine-tuning
    
    history_fine = model.fit(train_ds,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=val_ds)

    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    model_save_path = os.path.join(MODEL_DIR, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save class names list (Requirements: model/class_names.json)
    import json
    class_names_path = os.path.join(MODEL_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")

    # Save class indices dict (Optional but kept for reference)
    class_indices = {name: i for i, name in enumerate(class_names)}
    class_indices_path = os.path.join(MODEL_DIR, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {class_indices_path}")

def create_dummy_model():
    """Creates a dummy model structure for testing the app without a full dataset training run."""
    print("Creating dummy MobileNetV2 model for testing...")
    # This is just a placeholder if the user runs this without data
    # It creates the architecture but weights are random/imagenet default without training
    num_classes = 38 
    
    base_model = applications.MobileNetV2(input_shape=TARGET_IMG_SIZE + (3,),
                                          include_top=False,
                                          weights='imagenet')
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Resizing(TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1])(inputs)
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model.save(os.path.join(MODEL_DIR, 'model.h5'))
    print("Dummy model saved to model/model.h5")

if __name__ == '__main__':
    # Check if dataset exists
    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        train_model()
    else:
        print("Dataset not found or empty.")
        choice = input("Do you want to create a dummy model architecture for testing? (y/n): ")
        if choice.lower() == 'y':
            create_dummy_model()
