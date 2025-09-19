import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Flatten, Dense, Dropout
import joblib  # For saving and loading the scaler

# Load dataset
df = pd.read_csv('synthetic_strongly_correlated_dataset.csv')

# Handling non-numeric data by converting it to numeric or encoding it
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is non-numeric (categorical or string)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))  # Convert string values to numbers

# Assuming the label column is named 'y' and features are the rest
X = df.drop('y', axis=1).values  # Drop the 'y' column for features
y = df['y'].values  # Use the 'y' column as the target

# Ensure there are no missing values (fill NaNs with 0, or use an appropriate strategy)
X = pd.DataFrame(X).fillna(0).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Reshape the data for CNN and BiLSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for CNN
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)      # Reshape for CNN

# Build the hybrid model
model = Sequential()
# CNN layers
model.add(Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(Dropout(0.5))

# BiLSTM layer
model.add(Bidirectional(LSTM(50, return_sequences=False)))  # Use return_sequences=True if you have more LSTM layers

# Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Adjust for number of classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, verbose=1)

# Save the trained model
model.save('trained_model.h5')

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Print Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Optional: Plot loss and accuracy curves
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the predictions and features for later use
df['predicted_y'] = np.append(y_train, y_pred_classes)
df.to_csv('synthetic_strongly_correlated_dataset_with_predictions.csv', index=False)
