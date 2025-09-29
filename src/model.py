import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

class TrafficSignModel:
    def __init__(self, input_shape, num_classes=43):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build the CNN model architecture"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=self.input_shape),
            Conv2D(filters=64, kernel_size=(5,5), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(rate=0.15),
            
            # Second Convolutional Block
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(rate=0.20),
            
            # Classification Head
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(rate=0.25),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """Compile the model"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )
        
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is not None:
            return self.model.summary()
