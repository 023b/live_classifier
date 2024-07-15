import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import time

def create_project_structure():
    """Create the necessary directories for the webcam project."""
    directories = [
        'data',
        'data/training',
        'data/testing',
        'data/training/face',  # Special directory for face images
        'data/testing/face',
        'data/training/water_bottle',
        'data/testing/water_bottle',
        'data/training/remote',
        'data/testing/remote',
        'data/training/watch',
        'data/testing/watch',
        'data/training/background',  # For background/none class
        'data/testing/background',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_sample_images(class_name, num_images=30):
    """
    Generate placeholder sample images for a class.
    These are just to have some initial data structure.
    You'll replace these with real webcam captures.
    """
    # Create directory if it doesn't exist
    train_dir = os.path.join('data/training', class_name)
    test_dir = os.path.join('data/testing', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Choose a color for the class
    colors = {
        'face': (200, 150, 150),  # Skin tone-ish
        'water_bottle': (150, 200, 255),  # Blue-ish
        'remote': (100, 100, 100),  # Gray
        'watch': (200, 200, 100),  # Gold-ish
        'background': (50, 50, 50)  # Dark
    }
    
    base_color = colors.get(class_name, (np.random.randint(100, 200), 
                                         np.random.randint(100, 200), 
                                         np.random.randint(100, 200)))
    
    # Generate images
    all_files = []
    for i in range(num_images):
        # Create a unique filename
        filename = f"{class_name}_{i+1}.jpg"
        filepath = os.path.join('temp', filename)
        
        # Create directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Create a colored image
        width, height = 200 + i % 5 * 10, 200 + i % 3 * 10
        img = np.ones((height, width, 3), dtype=np.uint8)
        
        # Add some random variation to the color
        color = (
            max(0, min(255, base_color[0] + np.random.randint(-20, 20))),
            max(0, min(255, base_color[1] + np.random.randint(-20, 20))),
            max(0, min(255, base_color[2] + np.random.randint(-20, 20)))
        )
        
        # Fill the image
        img[:, :] = color
        
        # Add some patterns based on the class
        if class_name == 'face':
            # Simple face-like oval
            center_x, center_y = width // 2, height // 2
            axes_length = (width // 3, height // 2)
            cv2.ellipse(img, (center_x, center_y), axes_length, 0, 0, 360, (150, 150, 150), -1)
            
            # Eyes
            eye_radius = max(5, min(width, height) // 15)
            left_eye_x = center_x - width // 6
            right_eye_x = center_x + width // 6
            eyes_y = center_y - height // 10
            cv2.circle(img, (left_eye_x, eyes_y), eye_radius, (255, 255, 255), -1)
            cv2.circle(img, (right_eye_x, eyes_y), eye_radius, (255, 255, 255), -1)
            
            # Mouth
            mouth_width = width // 4
            mouth_height = height // 15
            mouth_y = center_y + height // 6
            cv2.ellipse(img, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (100, 100, 100), -1)
            
        elif class_name == 'water_bottle':
            # Simple water bottle shape
            bottle_width = width // 3
            bottle_height = int(height * 0.8)
            bottle_x = (width - bottle_width) // 2
            bottle_y = (height - bottle_height) // 2
            cv2.rectangle(img, (bottle_x, bottle_y), 
                         (bottle_x + bottle_width, bottle_y + bottle_height), 
                         (0, 0, 200), -1)
            
            # Bottle cap
            cap_width = bottle_width // 2
            cap_height = bottle_height // 8
            cap_x = bottle_x + (bottle_width - cap_width) // 2
            cv2.rectangle(img, (cap_x, bottle_y), 
                         (cap_x + cap_width, bottle_y + cap_height), 
                         (150, 150, 150), -1)
            
            # Water level
            water_height = int(bottle_height * 0.7)
            water_y = bottle_y + bottle_height - water_height
            cv2.rectangle(img, (bottle_x, water_y), 
                         (bottle_x + bottle_width, bottle_y + bottle_height), 
                         (255, 200, 0), -1)
            
        elif class_name == 'remote':
            # Remote shape
            remote_width = width // 3
            remote_height = int(height * 0.8)
            remote_x = (width - remote_width) // 2
            remote_y = (height - remote_height) // 2
            cv2.rectangle(img, (remote_x, remote_y), 
                         (remote_x + remote_width, remote_y + remote_height), 
                         (50, 50, 50), -1)
            
            # Buttons
            button_size = remote_width // 4
            for r in range(3):
                for c in range(2):
                    btn_x = remote_x + c * (button_size + 5) + 5
                    btn_y = remote_y + r * (button_size + 5) + remote_height // 3
                    cv2.rectangle(img, (btn_x, btn_y), 
                                 (btn_x + button_size, btn_y + button_size), 
                                 (200, 200, 200), -1)
            
        elif class_name == 'watch':
            # Watch face (circle)
            center_x, center_y = width // 2, height // 2
            watch_radius = min(width, height) // 3
            cv2.circle(img, (center_x, center_y), watch_radius, (200, 200, 200), -1)
            cv2.circle(img, (center_x, center_y), watch_radius, (0, 0, 0), 2)
            
            # Watch hands
            # Hour hand
            hour_angle = np.random.randint(0, 12) * 30 + np.random.randint(0, 30)
            hour_x = center_x + int(watch_radius * 0.5 * np.sin(np.radians(hour_angle)))
            hour_y = center_y - int(watch_radius * 0.5 * np.cos(np.radians(hour_angle)))
            cv2.line(img, (center_x, center_y), (hour_x, hour_y), (0, 0, 0), 2)
            
            # Minute hand
            minute_angle = np.random.randint(0, 60) * 6
            minute_x = center_x + int(watch_radius * 0.8 * np.sin(np.radians(minute_angle)))
            minute_y = center_y - int(watch_radius * 0.8 * np.cos(np.radians(minute_angle)))
            cv2.line(img, (center_x, center_y), (minute_x, minute_y), (0, 0, 0), 2)
            
            # Watch band
            band_width = watch_radius * 2
            band_height = watch_radius // 2
            cv2.rectangle(img, (center_x - band_width//2, center_y - watch_radius - band_height), 
                         (center_x + band_width//2, center_y - watch_radius), 
                         (150, 100, 50), -1)
            cv2.rectangle(img, (center_x - band_width//2, center_y + watch_radius), 
                         (center_x + band_width//2, center_y + watch_radius + band_height), 
                         (150, 100, 50), -1)
        
        # Save the image
        cv2.imwrite(filepath, img)
        all_files.append(filepath)
    
    return all_files

def split_files(files, class_name, train_ratio=0.8):
    """Split files into training and testing sets."""
    train_files, test_files = train_test_split(
        files, train_size=train_ratio, random_state=42
    )
    
    # Copy files to appropriate directories
    for src in train_files:
        dst = os.path.join('data/training', class_name, os.path.basename(src))
        shutil.copy(src, dst)
    
    for src in test_files:
        dst = os.path.join('data/testing', class_name, os.path.basename(src))
        shutil.copy(src, dst)
    
    return len(train_files), len(test_files)

def main():
    # Classes to generate sample images for
    classes = ['face', 'water_bottle', 'remote', 'watch', 'background']
    
    # Create project structure
    create_project_structure()
    
    # Generate and organize sample images
    for class_name in classes:
        print(f"Generating sample images for {class_name}...")
        files = generate_sample_images(class_name)
        num_train, num_test = split_files(files, class_name)
        print(f"Added {num_train} training and {num_test} testing images for {class_name}")
    
    # Clean up temporary files
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    
    print("\nProject setup complete!")
    print("Next steps:")
    print("1. Replace the sample images with real webcam captures using webcam_classifier.py")
    print("2. Run 'python webcam_classifier.py' to start the webcam classifier")

def capture_from_webcam():
    """Interactive function to capture images for each class from webcam."""
    classes = ['face', 'water_bottle', 'remote', 'watch', 'background']
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Please check your camera connection.")
        return
    
    try:
        for class_name in classes:
            print(f"\n--- Capturing images for {class_name} ---")
            print("Position the object in the frame.")
            print("Press 's' to start capturing, 'n' to skip this class, 'q' to quit")
            
            # Wait for user input
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image from webcam.")
                    break
                
                # Display instruction
                cv2.putText(frame, f"Press 's' to capture images for '{class_name}'", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Capture Training Images', frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit entirely
                    print("Quitting capture process.")
                    return
                elif key == ord('n'):  # Next class
                    print(f"Skipping {class_name}")
                    break
                elif key == ord('s'):  # Start capturing
                    # Create directories
                    train_dir = os.path.join('data/training', class_name)
                    test_dir = os.path.join('data/testing', class_name)
                    os.makedirs(train_dir, exist_ok=True)
                    os.makedirs(test_dir, exist_ok=True)
                    
                    # Capture images
                    num_images = 30  # Total to capture
                    train_count = int(num_images * 0.8)  # 80% for training
                    test_count = num_images - train_count  # 20% for testing
                    
                    print(f"Capturing {train_count} training images and {test_count} testing images...")
                    
                    # Capture training images
                    for i in range(train_count):
                        # Small delay for positioning
                        time.sleep(0.5)
                        
                        # Capture frame
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to capture image from webcam.")
                            continue
                        
                        # Save the image
                        filename = os.path.join(train_dir, f"{class_name}_train_{i+1}.jpg")
                        cv2.imwrite(filename, frame)
                        
                        # Display feedback
                        cv2.putText(frame, f"Captured training image {i+1}/{train_count}", 
                                  (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Capture Training Images', frame)
                        cv2.waitKey(100)  # Short display time
                    
                    print(f"Captured {train_count} training images.")
                    time.sleep(1)  # Pause before testing set
                    
                    # Capture testing images
                    for i in range(test_count):
                        # Small delay for positioning
                        time.sleep(0.5)
                        
                        # Capture frame
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to capture image from webcam.")
                            continue
                        
                        # Save the image
                        filename = os.path.join(test_dir, f"{class_name}_test_{i+1}.jpg")
                        cv2.imwrite(filename, frame)
                        
                        # Display feedback
                        cv2.putText(frame, f"Captured testing image {i+1}/{test_count}", 
                                  (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Capture Training Images', frame)
                        cv2.waitKey(100)  # Short display time
                    
                    print(f"Captured {test_count} testing images.")
                    break  # Move to next class
            
        print("\nImage capture complete!")
    
    finally:
        # Release webcam
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ask user if they want to use webcam or generate sample images
    print("Do you want to capture images from webcam or generate placeholder images?")
    print("1. Capture from webcam (recommended)")
    print("2. Generate placeholder images")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Create project structure first
        create_project_structure()
        # Then capture from webcam
        capture_from_webcam()
    else:
        # Just generate placeholder images
        main()