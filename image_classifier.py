import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SimpleImageClassifier:
    def __init__(self, k=5):
        """
        Initialize the image classifier.
        
        Parameters:
        k (int): Number of neighbors for KNN algorithm
        """
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.class_names = None
    
    def extract_features(self, image_path):
        """
        Extract simple features from an image.
        
        Parameters:
        image_path (str): Path to the image file
        
        Returns:
        numpy.ndarray: Feature vector
        """
        try:
            # Load and resize the image
            img = Image.open(image_path)
            img = img.resize((50, 50))  # Resize for consistency
            img_array = np.array(img)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            # Handle RGBA images (just use RGB channels)
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Flatten the image array to create a feature vector
            features = img_array.flatten()
            
            # Normalize features to be between 0 and 1
            features = features / 255.0
            
            return features
        
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """
        Load images from a directory structure organized by class.
        
        Parameters:
        data_dir (str): Path to the data directory
        
        Returns:
        tuple: (features, labels, class_names)
        """
        features = []
        labels = []
        class_names = []
        
        # Get all subdirectories (each is a class)
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                print(f"Loading class: {class_name}")
                
                # Process each image in the class directory
                for image_name in os.listdir(class_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_dir, image_name)
                        
                        # Extract features
                        image_features = self.extract_features(image_path)
                        
                        if image_features is not None:
                            features.append(image_features)
                            labels.append(class_name)
        
        return np.array(features), np.array(labels), class_names
    
    def train(self, train_dir):
        """
        Train the classifier on images in the training directory.
        
        Parameters:
        train_dir (str): Path to the training data directory
        
        Returns:
        float: Training accuracy
        """
        # Load the training data
        print(f"Loading training data from {train_dir}...")
        X, y, self.class_names = self.load_dataset(train_dir)
        
        if len(X) == 0:
            raise ValueError("No training data found")
        
        # Train the model
        print(f"Training model with {len(X)} images in {len(self.class_names)} classes...")
        self.model.fit(X, y)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Training accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def predict(self, image_path):
        """
        Predict the class of a single image.
        
        Parameters:
        image_path (str): Path to the image
        
        Returns:
        str: Predicted class name
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract features
        features = self.extract_features(image_path)
        
        if features is None:
            return None
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        return prediction
    
    def evaluate(self, test_dir):
        """
        Evaluate the model on a test dataset.
        
        Parameters:
        test_dir (str): Path to the test data directory
        
        Returns:
        tuple: (accuracy, predictions, true_labels)
        """
        # Load the test data
        print(f"Loading test data from {test_dir}...")
        X_test, y_test, _ = self.load_dataset(test_dir)
        
        if len(X_test) == 0:
            raise ValueError("No test data found")
        
        # Make predictions
        print(f"Making predictions on {len(X_test)} test images...")
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.2%}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the confusion matrix
        plt.savefig('results/confusion_matrix.png')
        print("Confusion matrix saved to results/confusion_matrix.png")
        
        return accuracy, y_pred, y_test
    
    def visualize_prediction(self, image_path):
        """
        Visualize a prediction on a single image.
        
        Parameters:
        image_path (str): Path to the image
        
        Returns:
        str: Predicted class name
        """
        # Predict the class
        prediction = self.predict(image_path)
        
        if prediction is None:
            return None
        
        # Display the image with prediction
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {prediction}")
        plt.axis('off')
        
        # Save the visualization
        result_path = f"results/prediction_{os.path.basename(image_path)}"
        plt.savefig(result_path)
        plt.close()
        
        print(f"Prediction visualization saved to {result_path}")
        
        return prediction
    
    def save_model(self, model_path="models/image_classifier.pkl"):
        """
        Save the trained model to a file.
        
        Parameters:
        model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model and class names
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.class_names, self.k), f)
        
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path="models/image_classifier.pkl"):
        """
        Load a trained model from a file.
        
        Parameters:
        model_path (str): Path to the saved model
        
        Returns:
        SimpleImageClassifier: Loaded classifier
        """
        with open(model_path, 'rb') as f:
            model, class_names, k = pickle.load(f)
        
        # Create a new classifier instance
        classifier = cls(k=k)
        classifier.model = model
        classifier.class_names = class_names
        
        print(f"Model loaded from {model_path}")
        print(f"Model can classify: {class_names}")
        
        return classifier