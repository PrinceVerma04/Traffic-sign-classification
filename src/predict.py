import numpy as np
import tensorflow as tf
from PIL import Image

class TrafficSignPredictor:
    def __init__(self, model_path, img_size=(30, 30)):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.class_names = [f'Class_{i}' for i in range(43)]  # Update with actual class names
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        image = Image.open(image_path)
        image = image.resize(self.img_size)
        image = np.array(image)
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    
    def predict(self, image_path):
        """Predict traffic sign class"""
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return {
            'class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': predictions[0]
        }
