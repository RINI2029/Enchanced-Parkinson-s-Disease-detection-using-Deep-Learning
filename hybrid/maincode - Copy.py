import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import xgboost as xgb
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# --- Image Classification Models and Parameters ---
WIDTH, HEIGHT, CHANNEL = 224, 224, 3

def process_single_image(img_path, width, height):
    """
    Loads and preprocesses a single image for prediction.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (width, height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

try:
    feature_extractor = load_model("cnn_feature_extractor.h5", compile=False)
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model("xgb_classifier_model.json")
    base_dir = r"D:\ss\2025\code\parkino\draw\dataset/"
    train_dir = os.path.join(base_dir, "train")
    
    if not os.path.isdir(train_dir) or not os.listdir(train_dir):
        messagebox.showerror("Error", f"The training directory '{train_dir}' is empty or does not exist. Cannot determine class names.")
        exit()
        
    image_target_names = sorted(os.listdir(train_dir))
    print("Image models and class names loaded successfully.")
except Exception as e:
    messagebox.showerror("Error", f"Error loading image models: {e}")
    exit()

# --- EEG Data Classification Models and Parameters ---
try:
    eeg_model = load_model('trained_model.h5')
    eeg_scaler = joblib.load('scaler.pkl')
    print("EEG models and scaler loaded successfully.")
except Exception as e:
    messagebox.showerror("Error", f"Error loading EEG models: {e}")
    exit()

# --- Hybrid GUI Application Class ---
class HybridClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid Parkinsons & EEG Classifier")
        self.geometry("800x700")
        self.image_path = None
        self.eeg_path = None
        self.image_prediction = None
        self.eeg_prediction = None
        self.create_widgets()

    def create_widgets(self):
        self.image_frame = tk.LabelFrame(self, text="Image Classification", padx=10, pady=10)
        self.image_frame.pack(pady=10, fill="x")

        self.browse_image_button = tk.Button(self.image_frame, text="Browse Image", command=self.browse_image)
        self.browse_image_button.pack(side=tk.LEFT, padx=10)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.image_result_label = tk.Label(self.image_frame, text="Please select an image.", font=("Arial", 12))
        self.image_result_label.pack(side=tk.LEFT, padx=10)

        tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN).pack(fill="x", padx=5, pady=5)

        self.eeg_frame = tk.LabelFrame(self, text="EEG Data Classification", padx=10, pady=10)
        self.eeg_frame.pack(pady=10, fill="x")

        self.browse_eeg_button = tk.Button(self.eeg_frame, text="Browse EEG Data", command=self.browse_eeg_data, state=tk.DISABLED)
        self.browse_eeg_button.pack(side=tk.LEFT, padx=10)

        self.eeg_result_label = tk.Label(self.eeg_frame, text="Browse EEG data after image classification.", font=("Arial", 12))
        self.eeg_result_label.pack(side=tk.LEFT, padx=10)

        tk.Frame(self, height=2, bd=1, relief=tk.SUNKEN).pack(fill="x", padx=5, pady=5)

        self.final_frame = tk.LabelFrame(self, text="Final Diagnosis", padx=10, pady=10)
        self.final_frame.pack(pady=10, fill="x")
        self.final_result_label = tk.Label(self.final_frame, text="Please classify both the image and EEG data.", font=("Arial", 16, "bold"))
        self.final_result_label.pack(pady=20)

    def browse_image(self):
        self.image_prediction = None
        self.eeg_prediction = None
        self.final_result_label.config(text="Please classify both the image and EEG data.")
        
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.image_path = file_path
            img = Image.open(self.image_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.image_result_label.config(text=f"Image Selected: {os.path.basename(file_path)}")
            self.classify_image()
            self.browse_eeg_button.config(state=tk.NORMAL)
            self.eeg_result_label.config(text="Now, please select your EEG data (CSV file).")
            self.final_result_label.config(text="Image classified. Awaiting EEG data to provide final diagnosis.")

    def classify_image(self):
        if not self.image_path:
            return
        
        processed_image = process_single_image(self.image_path, WIDTH, HEIGHT)
        if processed_image is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        features = feature_extractor.predict(processed_image, verbose=0)
        pred_proba = xgb_classifier.predict_proba(features)
        predicted_class_idx = np.argmax(pred_proba, axis=1)[0]
        predicted_label = image_target_names[predicted_class_idx]
        confidence = pred_proba[0][predicted_class_idx]
        
        self.image_prediction = predicted_label
        print(f"Predicted Image Class: {self.image_prediction}")  # Debugging output
        self.image_result_label.config(text=f"Predicted Image Class: {predicted_label}\nConfidence: {confidence:.2f}")

    def browse_eeg_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.eeg_path = filepath
            try:
                new_data = pd.read_csv(filepath)
                
                # Encoding categorical variables if any
                for column in new_data.columns:
                    if new_data[column].dtype == 'object':
                        le = LabelEncoder()
                        new_data[column] = le.fit_transform(new_data[column].astype(str))
                
                new_data = new_data.fillna(0).values
                new_data_scaled = eeg_scaler.transform(new_data)
                new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)
                
                # Predict EEG classes
                predictions = eeg_model.predict(new_data_scaled)
                predicted_classes = predictions.argmax(axis=1)
                print(f"Predicted EEG Classes: {predicted_classes}")  # Debugging output
                
                self.eeg_prediction = int(Counter(predicted_classes).most_common(1)[0][0])
                self.eeg_result_label.config(text=f"EEG Data Predicted Class: {self.eeg_prediction}")
                
                self.make_final_diagnosis()  # Fixed method call

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during EEG classification: {str(e)}")
                self.eeg_prediction = None
                self.final_result_label.config(text="EEG data classification failed. Please try again.")

    def make_final_diagnosis(self):
        if self.image_prediction is None or self.eeg_prediction is None:
            return

        print(f"Image Prediction: {self.image_prediction}, EEG Prediction: {self.eeg_prediction}")  # Debugging output
        final_diagnosis_text = "Diagnosis cannot be determined. Data conflict."
        
        # Clean the image prediction and check for Parkinson's disease
        image_is_parkinsons = "parkinson" in self.image_prediction.lower().strip()  # Case-insensitive match
        print(f"Image Prediction: {self.image_prediction}, Parkinsons detection: {image_is_parkinsons}")  # Debugging output
        
        # Logic for Parkinson's diagnosis based on image and EEG predictions
        if image_is_parkinsons:
            print(f"EEG Prediction Value: {self.eeg_prediction}")  # Debugging output
            if self.eeg_prediction >= 3:
                final_diagnosis_text = "Final Diagnosis: Severe Parkinsons (Confirmed by Image and EEG)"
            elif self.eeg_prediction == 2:
                final_diagnosis_text = "Final Diagnosis: Moderate Parkinsons"
            elif self.eeg_prediction < 2:
                final_diagnosis_text = "Final Diagnosis: Mild Parkinsons"
        
        elif "healthy" in self.image_prediction.lower().strip():
            # Assuming 'healthy' label from image classification
            if self.eeg_prediction in [0, 1]:
                final_diagnosis_text = "Final Diagnosis: Healthy (Confirmed by Image and EEG)"
            elif self.eeg_prediction in [2, 3]:
                final_diagnosis_text = "Final Diagnosis: Mild Condition (Image is healthy, but EEG suggests mild issue)"
            elif self.eeg_prediction > 3:
                final_diagnosis_text = "Final Diagnosis: Severe Condition (Image is healthy, but EEG suggests a severe issue)"
        
        print(f"Final Diagnosis: {final_diagnosis_text}")  # Debugging output
        self.final_result_label.config(text=final_diagnosis_text)

if __name__ == "__main__":
    app = HybridClassifierApp()
    app.mainloop()
