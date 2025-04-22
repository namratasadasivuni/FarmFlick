import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import os
from tkinter import filedialog, Tk

# Load model
model = tf.keras.models.load_model("../model/crop_model.h5")
model = tf.keras.models.load_model(r"C:\Users\Namrata\Desktop\CSIHACK\FarmFlick\model\crop_model.h5")

# Load class names from dataset folder
dataset_dir = "C:/Users/Namrata/Desktop/CSIHACK/FarmFlick/dataset"
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

# Initialize webcam
cap = cv2.VideoCapture(0)
streaming = False

def toggle_stream():
    global streaming
    streaming = not streaming

def preprocess_and_predict(image):
    img = cv2.resize(image, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    idx = np.argmax(prediction)
    label = class_names[idx]
    confidence = np.max(prediction)
    return label, confidence

def predict_uploaded_image():
    # Open file dialog to upload image
    root = Tk()
    root.withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        label, confidence = preprocess_and_predict(image)
        display_text = f"{label.capitalize()} ({confidence:.2f})"
        cv2.putText(image, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Show the uploaded image with the prediction
        cv2.imshow("Uploaded Image Prediction", image)

        while True:
            if cv2.getWindowProperty("Uploaded Image Prediction", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

def capture_and_predict():
    ret, frame = cap.read()
    if ret:
        label, confidence = preprocess_and_predict(frame)
        display_text = f"{label.capitalize()} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Show the captured frame with the prediction
        cv2.imshow("Captured Image Prediction", frame)

        while True:
            if cv2.getWindowProperty("Captured Image Prediction", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

print("Press 's' to start/stop live stream")
print("Press 'u' to upload an image for prediction")
print("Press 'c' to capture image from webcam for prediction")
print("Press 'q' to quit")

while True:
    if streaming:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        label, confidence = preprocess_and_predict(frame)
        display_text = f"{label.capitalize()} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the live stream frame with the prediction
        cv2.imshow("Live Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        toggle_stream()

    elif key == ord('u'):
        predict_uploaded_image()

    elif key == ord('c'):
        capture_and_predict()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
