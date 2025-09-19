import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Function to browse and load CSV file
def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        try:
            # Load the selected CSV file
            new_data = pd.read_csv(filepath)
            
            # Preprocess the new data (handle missing values, encode categorical data)
            for column in new_data.columns:
                if new_data[column].dtype == 'object':
                    le = LabelEncoder()
                    new_data[column] = le.fit_transform(new_data[column].astype(str))
            
            new_data = new_data.fillna(0).values  # Handle missing values

            # Scale the data using the saved scaler
            new_data_scaled = scaler.transform(new_data)

            # Reshape the data for model input
            new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)

            # Predict using the trained model
            predictions = model.predict(new_data_scaled)
            predicted_classes = predictions.argmax(axis=1)  # Convert probabilities to class labels
            
            # Display the predictions in the results box
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Predicted Classes:\n{predicted_classes}")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Set up the GUI window
root = tk.Tk()
root.title("Classify Data with Trained Model")
root.geometry("500x400")

# Add a button to browse files
browse_button = tk.Button(root, text="Browse CSV File", command=browse_file)
browse_button.pack(pady=20)

# Add a Text box to display predictions
result_text = tk.Text(root, height=10, width=60)
result_text.pack(pady=20)

# Start the GUI event loop
root.mainloop()
