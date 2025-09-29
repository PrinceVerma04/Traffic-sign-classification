import argparse
from src.predict import TrafficSignPredictor

def main():
    parser = argparse.ArgumentParser(description='Predict Traffic Sign from Image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image for prediction')
    
    args = parser.parse_args()
    
    predictor = TrafficSignPredictor(args.model_path)
    result = predictor.predict(args.image_path)
    
    print(f"Predicted Class: {result['class']} ({result['class_name']})")
    print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
