import tensorflow as tf
import datetime
from .utils import plot_performance

class ModelTrainer:
    def __init__(self, model, model_save_path='models/traffic_sign_model.h5'):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
    
    def train(self, X_train, y_train, X_test, y_test, epochs=35, batch_size=128):
        """Train the model"""
        print("Starting model training...")
        
        # Train with GPU if available
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=1
            )
        
        # Save the trained model
        self.model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available!")
        
        plot_performance(self.history, save_path)
