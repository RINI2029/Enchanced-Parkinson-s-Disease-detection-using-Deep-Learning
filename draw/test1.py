import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import xgboost as xgb
from tensorflow.keras.models import load_model
import tensorflow as tf

# Set up directories and parameters
WIDTH, HEIGHT, CHANNEL = 224, 224, 3

# Define image processing function
def process_single_image(img_path, width, height):
    """
    Loads and preprocesses a single image for prediction.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (width, height))
    img = img / 255.0
    # Add a batch dimension to the image
    img = np.expand_dims(img, axis=0)
    return img

# Load the pre-trained models
try:
    feature_extractor = load_model("cnn_feature_extractor.h5", compile=False)
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model("xgb_classifier_model.json")
    
    # Get class names from the dataset directory
    base_dir = r"D:\ss\2025\code\parkino\draw\dataset/"
    train_dir = os.path.join(base_dir, "train")
    target_names = sorted(os.listdir(train_dir))
    NUM_CLASSES = len(target_names)
    print("Models and class names loaded successfully.")

except Exception as e:
    messagebox.showerror("Error", f"Error loading models: {e}")
    exit()

class ImageClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Classifier")
        self.geometry("800x600")
        self.image_path = None
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        # Browse button
        browse_button = tk.Button(button_frame, text="Browse Image", command=self.browse_image)
        browse_button.pack(side=tk.LEFT, padx=10)
        
        # Classify button
        classify_button = tk.Button(button_frame, text="Classify Image", command=self.classify_image)
        classify_button.pack(side=tk.LEFT, padx=10)

        # Label to display image
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        # Label to display results
        self.result_label = tk.Label(self, text="Please select an image to classify.", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            # Display the selected image
            img = Image.open(self.image_path)
            img.thumbnail((400, 400)) # Resize image to fit in the window
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.result_label.config(text=f"Selected: {os.path.basename(file_path)}")

    def classify_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Process the input image
        processed_image = process_single_image(self.image_path, WIDTH, HEIGHT)
        if processed_image is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        # Extract features using the CNN model
        features = feature_extractor.predict(processed_image, verbose=0)

        # Classify using the XGBoost model
        pred_proba = xgb_classifier.predict_proba(features)
        predicted_class_idx = np.argmax(pred_proba, axis=1)[0]
        predicted_label = target_names[predicted_class_idx]
        confidence = pred_proba[0][predicted_class_idx]
        
        # Update the result label
        self.result_label.config(text=f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}")

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
