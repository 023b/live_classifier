import argparse
import os
import sys
from webcam_classifier import WebcamClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Webcam Object Classifier')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup the project and capture/generate sample data')
    setup_parser.add_argument('--generate', action='store_true', 
                             help='Generate placeholder images instead of capturing from webcam')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the webcam classifier')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture training images for a new class')
    capture_parser.add_argument('class_name', type=str, help='Name of the class to capture')
    capture_parser.add_argument('--num', type=int, default=20, 
                               help='Number of images to capture (default: 20)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the classifier on existing data')
    
    return parser.parse_args()

def setup(generate=False):
    """Run the setup script."""
    import setup_webcam
    
    if generate:
        setup_webcam.main()
    else:
        # Create project structure first
        setup_webcam.create_project_structure()
        # Then capture from webcam
        setup_webcam.capture_from_webcam()

def run():
    """Run the webcam classifier."""
    model_path = 'models/webcam_classifier.pkl'
    
    # Check if model exists, train if it doesn't
    if not os.path.exists(model_path):
        print("No trained model found. Training model first...")
        classifier = WebcamClassifier()
        if os.path.exists('data/training'):
            classifier.classifier.train('data/training')
            classifier.classifier.save_model(model_path)
        else:
            print("No training data found. Please run setup first.")
            return
    
    # Run the classifier
    classifier = WebcamClassifier(model_path)
    classifier.run()

def capture(class_name, num_images=20):
    """Capture training images for a new class."""
    classifier = WebcamClassifier()
    classifier.capture_training_images(class_name, num_images)

def train():
    """Train the classifier on existing data."""
    classifier = WebcamClassifier()
    if os.path.exists('data/training'):
        classifier.classifier.train('data/training')
        classifier.classifier.save_model('models/webcam_classifier.pkl')
        print("Model trained and saved.")
    else:
        print("No training data found. Please run setup first.")

def main():
    """Main function to run the appropriate command."""
    args = parse_args()
    
    if args.command == 'setup':
        setup(args.generate if hasattr(args, 'generate') else False)
    elif args.command == 'run':
        run()
    elif args.command == 'capture':
        capture(args.class_name, args.num)
    elif args.command == 'train':
        train()
    else:
        # If no command is provided, show help
        print("Please specify a command. Use --help for more information.")
        print("\nAvailable commands:")
        print("  setup    - Setup the project and capture/generate sample data")
        print("  run      - Run the webcam classifier")
        print("  capture  - Capture training images for a new class")
        print("  train    - Train the classifier on existing data")

if __name__ == "__main__":
    main()