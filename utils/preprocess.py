import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image for the model.
    Steps:
    1. Open image.
    2. Resize to target_size.
    3. Convert to numpy array.
    4. Normalize pixel values (0-1).
    5. Expand dimensions to match model input shape (batch_size, height, width, channels).
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Ensure 3 channels (RGB)
        if img_array.shape[-1] != 3:
             img_array = img_array[..., :3] # basic handling for alpha channel

        img_array = img_array.astype('float32')
        # img_array = img_array / 255.0  # Removed: Model has Rescaling layer
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
