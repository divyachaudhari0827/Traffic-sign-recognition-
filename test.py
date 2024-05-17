import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sys

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load the pre-trained model
model = load_model("C:\\Users\\HP\\Desktop\\traffic\\Traffic_Signs_WebApp-master\\model\\model.h5")

# Function to preprocess the input image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Dictionary mapping class indices to human-readable labels
class_labels = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    # Add more class labels as per your dataset
}

# Function to predict traffic sign from image
def predict_traffic_sign(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    class_index = np.argmax(prediction)
    raw_predictions = prediction[0]
    print("Raw Predictions:", raw_predictions)
    return class_labels.get(class_index, "Unknown")

# Function to capture image from webcam and perform traffic sign recognition
def capture_and_recognize_traffic_sign():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Failed to open webcam")
        return
    
    print("Traffic sign recognition started. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        # Display the captured frame
        cv2.imshow("Traffic Sign Recognition", frame)
        
        # Preprocess the frame and predict the traffic sign
        traffic_sign = predict_traffic_sign(frame)
        
        # Display the predicted traffic sign label
        print("Predicted Traffic Sign:", traffic_sign)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Perform real-time traffic sign recognition
capture_and_recognize_traffic_sign()
