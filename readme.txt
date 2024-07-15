# Webcam Object Classifier

An extension of the simple image classifier that allows real-time object classification using your webcam. This system can identify objects like faces, water bottles, remotes, watches, and more.

## Features

- Real-time webcam classification
- Easy training interface - capture new classes directly through the webcam
- Multiple recognition classes
- Confidence indicators
- Simple and intuitive interface

## Requirements

```
numpy
pillow
sklearn
matplotlib
opencv-python
```

Install dependencies with:

```bash
pip install numpy pillow scikit-learn matplotlib opencv-python
```

## Quick Start

1. Set up the project and capture images:

```bash
python main_webcam.py setup
```

2. Run the webcam classifier:

```bash
python main_webcam.py run
```

## Project Structure

```
.
├── data/
│   ├── training/          # Training images organized by class
│   │   ├── face/          # Your face images
│   │   ├── water_bottle/  # Water bottle images
│   │   ├── remote/        # Remote control images
│   │   ├── watch/         # Watch images
│   │   └── background/    # Background/none class images
│   └── testing/           # Testing images organized by class
├── models/                # Saved model files
├── results/               # Prediction visualizations
├── setup_webcam.py        # Script to set up project and capture/generate images
├── webcam_classifier.py   # Webcam interface and classifier
├── main_webcam.py         # Command-line interface
└── README_WEBCAM.md
```

## How It Works

This system extends the simple image classifier with webcam capabilities:

1. **Image Capture**: Uses OpenCV to access your webcam and capture frames
2. **Feature Extraction**: Each frame is processed and converted to a feature vector
3. **Classification**: The model predicts what object is in the frame
4. **Visualization**: Shows the prediction and confidence level in real-time

## Commands

### Setup
```bash
# Capture images from webcam (recommended)
python main_webcam.py setup

# Generate placeholder images instead
python main_webcam.py setup --generate
```

### Run the Classifier
```bash
python main_webcam.py run
```

### Capture Additional Training Images
```bash
# Capture 20 images of a new class called "keyboard"
python main_webcam.py capture keyboard

# Capture 30 images instead
python main_webcam.py capture keyboard --num 30
```

### Train the Model
```bash
# Retrain the model if you've added new images manually
python main_webcam.py train
```

## Using the Webcam Interface

When running the webcam classifier:

- Press 'q' to quit the application
- Press 'c' to capture images for a new class
- The prediction and confidence level are displayed on screen
- Higher confidence is shown in green, medium in yellow, and low in red

## Adding Your Face Images

The setup script creates a special directory for face images. During setup:

1. When prompted to capture images for the "face" class, position your face in the frame
2. The system will capture multiple images of your face from different angles
3. These images will be used to train the classifier to recognize your face

## Tips for Best Results

- **Lighting**: Ensure good, consistent lighting for both training and classification
- **Positioning**: Try to position objects similarly during training and testing
- **Background**: Use a consistent background for better results
- **Variety**: Capture images from different angles for more robust recognition
- **Quantity**: Add more training images if an object is not being recognized well

## Extending the Project

- Add more object classes by capturing new training data
- Improve the feature extraction for better accuracy
- Add tracking capabilities to follow objects in the frame
- Implement a graphical user interface for easier interaction

## Troubleshooting

- **Webcam not detected**: Ensure your webcam is properly connected and not in use by another application
- **Poor recognition**: Try adding more training images from different angles and lighting conditions
- **Model loading error**: Make sure you've run the setup and training commands first
- **OpenCV errors**: Ensure you have the correct version of OpenCV installed for your system