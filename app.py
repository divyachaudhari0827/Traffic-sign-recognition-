from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Classes of traffic signs
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicle > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons' }



# Load the model
model_file_path = 'C:\\Users\\HP\\Desktop\\traffic\\Traffic_Signs_WebApp-master\\model\\model.h5'

if os.path.exists(model_file_path):
    print("Model file exists at:", model_file_path)
    try:
        model = load_model(model_file_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading the model:", e)
else:
    print("Model file not found at:", model_file_path)


import cv2

def preprocessing(img):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image to model's input size
    img_resized = cv2.resize(img_gray, (32, 32))
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    # Add channel dimension
    img_processed = img_normalized.reshape((1, 32, 32, 1))
    return img_processed



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            file_name = secure_filename(f.filename)
            upload_dir = 'uploads'
            file_path = os.path.join(upload_dir, file_name)

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            f.save(file_path)

            if os.path.exists(file_path):
                try:
                    # Preprocess the uploaded image
                    img = cv2.imread(file_path)
                    img_processed = preprocessing(img)

                    # Make prediction using the preprocessed image
                    predicted_probabilities = model.predict(img_processed)
                    predicted_class_index = np.argmax(predicted_probabilities)
                    predicted_class_label = classes[predicted_class_index]

                    result = "Predicted Traffic Sign is: " + predicted_class_label
                except Exception as e:
                    result = "Error occurred while processing the image: " + str(e)
            else:
                result = "Error: Uploaded file not found at the specified path."
        except Exception as e:
            result = "Error occurred: " + str(e)

        # Return the result as JSON
        return jsonify({'result': result})

    return "Error: Method Not Allowed. Please use POST method."


if __name__ == '__main__':
    app.run(debug=True)
