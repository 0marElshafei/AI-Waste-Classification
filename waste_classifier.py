# -*- coding: utf-8 -*-
"""
AI Waste Classification — Computer Vision Garbage Classifier

Originally built during the University of Toronto CREATE "AI: Sorting it Out!"
summer design program. Classifies waste items in real time from a webcam feed
using a Teachable Machine-trained Keras model, and maps each prediction to the
correct disposal bin (Landfill, Organic, Paper Recycling, or Container Recycling).

Model paths below are relative to this file, so the script runs as-is once
keras_model.h5 and labels.txt are in the same directory.
"""

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Import opencv-python
import h5py  # Import h5py to read HDF5 files
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# --- Load Model and Labels ---
# Reformat the .h5 file provided by Teachable Machine to ensure compatibility
# with newer versions of Python and TensorFlow.
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

f = h5py.File(MODEL_PATH, mode="r+")

model_config_string = f.attrs.get("model_config")

if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')

f.attrs.modify('model_config', model_config_string)
f.flush()

model_config_string = f.attrs.get("model_config")

assert model_config_string.find('"groups": 1,') == -1

# Load the model
model = load_model(MODEL_PATH, compile=False)
model.summary()  # Print the model summary to understand its architecture

# Load the labels
class_names = open(LABELS_PATH, "r").readlines()

# --- Display settings ---
font = cv2.FONT_HERSHEY_TRIPLEX  # Font type
fontScale = .5  # Font scale
color = (255, 255, 255)  # Text color
thickness = 1  # Thickness of the text

# --- Categorize Waste ---
# Clean class names: each entry in class_names is formatted like
# "integer class_name\n", so strip the index, leading space, and newline.
class_list = []

for entry in class_names:
    class_name = entry[2:-1]
    class_list.append(class_name)

# Map each waste class to its correct disposal bin.
waste_categorization = pd.DataFrame(class_list, columns=['Classification'])
waste_categorization = waste_categorization.set_index(['Classification'])
waste_categorization["Waste Bin"] = "Unknown"

waste_categorization.loc['Marker'] = 'Landfill'
waste_categorization.loc['Egg Carton'] = 'Paper Recycling'
waste_categorization.loc['Paper Plate'] = 'Paper Recycling'
waste_categorization.loc['Can'] = 'Container Recycling'
waste_categorization.loc['Tin Foil'] = 'Container Recycling'
waste_categorization.loc['Blue Folder'] = 'Paper Recycling'
waste_categorization.loc['Plastic Bottle'] = 'Container Recycling'
waste_categorization.loc['Stick'] = 'Organic'
waste_categorization.loc['Cardboard'] = 'Paper Recycling'
waste_categorization.loc['Plastic Spoon'] = 'Container  Recycling'
waste_categorization.loc[' Plastic Bag'] = 'Container Recycling'
waste_categorization.loc[' Mask'] = 'Landfill'


def predict_model(model, prediction_image):
    """Run the model on a single frame and return the predicted class and confidence."""
    image = prediction_image.copy()
    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1.0

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index][2:-1]
    confidence_score = prediction[0][index]
    return class_name, confidence_score


def get_waste_bin(class_list, class_name, confidence_score):
    """Return the correct waste bin for a prediction, but only above a confidence threshold."""
    waste_bin = ""  # Default to empty if confidence is too low

    if confidence_score > 0.67:
        waste_bin = class_list.at[class_name, "Waste Bin"]
    return waste_bin


def add_text_to_image(image, class_name, waste_bin, confidence_score):
    """Overlay the live prediction and disposal instructions on the camera frame."""
    instructions = "Place Item in Front of Camera"
    message_1 = f"{class_name}, {confidence_score}"
    message_2 = f'{waste_bin}'

    org_i = (50, 50)
    org_1 = (50, 100)
    org_2 = (50, 150)

    cv2.putText(image, instructions, org_i, font, fontScale, color, thickness)
    cv2.putText(image, message_1, org_1, font, fontScale, color, thickness)
    cv2.putText(image, message_2, org_2, font, fontScale, color, thickness)


def main():
    camera = cv2.VideoCapture(0)
    while True:
        # Grab the webcam's image.
        ret, image = camera.read()

        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

        class_name, confidence_score = predict_model(model, image)

        print("Class: ", class_name, end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        add_text_to_image(
            image,
            class_name,
            get_waste_bin(waste_categorization, class_name, confidence_score),
            confidence_score,
        )

        cv2.imshow("Waste Disposal Categorization Tool", image)

        # Close the window if the user presses the 'esc' key (ASCII 27).
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
