import os
import argparse
from src.data_loader import DataLoader
from src.model import TrafficSignModel
from src.train import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Traffic Sign Classification Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=35, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='models/traffic_sign_model.h5', 
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(args.data_path)
    data, labels = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.prepare_data()
    
    # Build model
    print("Building model...")
    model_builder = TrafficSignModel(input_shape=X_train.shape[1:])
    model = model_builder.compile_model()
    model_builder.summary()
    
    # Train model
    print("Starting training...")
    trainer = ModelTrainer(model, args.model_save_path)
    history = trainer.train(X_train, y_train, X_test, y_test, 
                          epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate model
    print("Evaluating model...")
    trainer.evaluate(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history('results')

if __name__ == "__main__":
    main()
