import numpy as np
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataLoader:
    def __init__(self, data_path, img_size=(30, 30), num_classes=43):
        self.data_path = data_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.data = []
        self.labels = []
    
    def load_data(self):
        """Load and preprocess the traffic sign data"""
        for i in range(self.num_classes):
            path = os.path.join(self.data_path, 'train', str(i))
            images = os.listdir(path)
            
            for img_name in images:
                try:
                    image = Image.open(os.path.join(path, img_name))
                    image = image.resize(self.img_size)
                    image = np.array(image)
                    self.data.append(image)
                    self.labels.append(i)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        return self.data, self.labels
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data and prepare for training"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )
        
        # Convert to categorical
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        return X_train, X_test, y_train, y_test
