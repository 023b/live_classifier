import cv2
import numpy as np
import os
import time
from image_classifier import SimpleImageClassifier

class WebcamClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the webcam classifier.
        
        Parameters:
        model_path (str): Path to a pre-trained model file
        """
        # Initialize the classifier
        if model_path and os.path.exists(model_path):
            self.classifier = SimpleImageClassifier.load_model(model_path)
        else:
            self.classifier = SimpleImageClassifier(k=5)
            if os.path.exists('data/training'):
                print("Training model on existing data...")
                self.classifier.train('data/training')
                self.classifier.save_model('models/webcam_classifier.pkl')
            else:
                print("No training data found. Please run setup_webcam.py first.")
        
        # Initialize webcam
        self.cap = None
        self.frame_skip = 10  # Process every 10th frame for efficiency
        self.frame_count = 0
        self.last_prediction = None
        self.prediction_confidence = 0
        
    def start_webcam(self):
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam. Please check your camera connection.")
        
        print("Webcam started. Press 'q' to quit, 'c' to capture training images.")
        
    def stop_webcam(self):
        """Release the webcam."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            print("Webcam stopped.")
    
    def capture_training_images(self, class_name, num_images=20, delay=0.5):
        """
        Capture training images from webcam for a specific class.
        
        Parameters:
        class_name (str): Name of the class to capture
        num_images (int): Number of images to capture
        delay (float): Delay between captures in seconds
        """
        if not self.cap or not self.cap.isOpened():
            self.start_webcam()
        
        # Create directory if it doesn't exist
        class_dir = os.path.join('data/training', class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Capturing {num_images} images for class '{class_name}'...")
        print("Position the object in the frame.")
        print("Press 's' to start capturing.")
        
        # Wait for 's' key to start
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                return
            
            # Display instruction
            cv2.putText(frame, f"Press 's' to start capturing for '{class_name}'", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Capture Training Images', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                self.stop_webcam()
                return
            elif key == ord('s'):  # Start capturing
                break
        
        # Capture images
        images_captured = 0
        while images_captured < num_images:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                continue
            
            # Save the image
            filename = os.path.join(class_dir, f"{class_name}_{time.time()}.jpg")
            cv2.imwrite(filename, frame)
            images_captured += 1
            
            # Display feedback
            cv2.putText(frame, f"Captured {images_captured}/{num_images} for '{class_name}'", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture Training Images', frame)
            
            # Wait for delay
            key = cv2.waitKey(int(delay * 1000)) & 0xFF
            if key == ord('q'):  # Quit
                break
        
        print(f"Captured {images_captured} images for class '{class_name}'.")
        
        # Train the model with the new data
        print("Retraining model with new data...")
        self.classifier.train('data/training')
        self.classifier.save_model('models/webcam_classifier.pkl')
        
        print("Training complete. Ready for classification.")
    
    def process_frame(self, frame):
        """
        Process a single frame for classification.
        
        Parameters:
        frame (numpy.ndarray): Image frame from webcam
        
        Returns:
        str: Predicted class name
        """
        # Save frame to temporary file
        temp_file = 'temp_frame.jpg'
        cv2.imwrite(temp_file, frame)
        
        # Predict class
        prediction = self.classifier.predict(temp_file)
        
        # Update confidence
        if prediction == self.last_prediction:
            self.prediction_confidence += 0.2
            self.prediction_confidence = min(self.prediction_confidence, 1.0)
        else:
            self.prediction_confidence = 0.5
        
        self.last_prediction = prediction
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return prediction
    
    def run(self):
        """Run the webcam classifier loop."""
        if not self.cap or not self.cap.isOpened():
            self.start_webcam()
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                break
            
            # Process every nth frame
            if self.frame_count % self.frame_skip == 0:
                prediction = self.process_frame(frame)
                
                # Display prediction with confidence
                confidence_text = f"Confidence: {self.prediction_confidence:.1f}"
                
                # Choose color based on confidence
                if self.prediction_confidence >= 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif self.prediction_confidence >= 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Display prediction
                cv2.putText(frame, f"Prediction: {prediction}", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, confidence_text,
                          (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                          
                # Display instructions
                cv2.putText(frame, "Press 'q' to quit, 'c' to capture new class", 
                          (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Webcam Object Classifier', frame)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('c'):  # Capture new training data
                class_name = input("Enter class name for the object: ")
                self.capture_training_images(class_name)
        
        # Clean up
        self.stop_webcam()

if __name__ == "__main__":
    classifier = WebcamClassifier('models/webcam_classifier.pkl')
    classifier.run()