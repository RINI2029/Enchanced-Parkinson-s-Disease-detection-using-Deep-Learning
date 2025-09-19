import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Reshape, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set up directories and parameters
base_dir = r"D:\ss\2025\code\parkino\draw\dataset/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "val1")
WIDTH, HEIGHT, CHANNEL = 224, 224, 3
NUM_CLASSES = 0 # To be determined dynamically
EPOCHS = 50
VERBOSE = 1

# Define image processing function
def process_block(img, width, height):
    img = cv2.resize(img, (width, height))
    img = img / 255.0
    return img

# Prepare training data
X_train = []
y_train = []
train_classes = sorted(os.listdir(train_dir))
class_to_idx = {cls: i for i, cls in enumerate(train_classes)}

for dirname in train_classes:
    print(f"Processing training images in {dirname}")
    for file_name in os.listdir(os.path.join(train_dir, dirname)):
        img_path = os.path.join(train_dir, dirname, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = process_block(img, WIDTH, HEIGHT)
            X_train.append(img)
            y_train.append(class_to_idx[dirname])

# Prepare testing data
X_test = []
y_test = []

for dirname in sorted(os.listdir(test_dir)):
    print(f"Processing testing images in {dirname}")
    # Handle cases where a class might be in test but not train
    if dirname in class_to_idx:
        for file_name in os.listdir(os.path.join(test_dir, dirname)):
            img_path = os.path.join(test_dir, dirname, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = process_block(img, WIDTH, HEIGHT)
                X_test.append(img)
                y_test.append(class_to_idx[dirname])

NUM_CLASSES = len(class_to_idx)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("Unique labels after manual mapping (train):", np.unique(y_train))
print("Unique labels after manual mapping (test):", np.unique(y_test))

print(f"Shape of images in X_train: {X_train.shape}")
print(f"Shape of images in X_test: {X_test.shape}")

# One-hot encode the labels for the Keras model
y_train_onehot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

# Build the CNN-LSTM model
def build_cnn_lstm_model(input_shape, num_classes):
    img_input = Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Corrected Reshape and LSTM connection
    # Get the shape of the output from the last Conv2D layer
    cnn_output_shape = x.shape
    
    # Reshape to a 3D tensor: (batch_size, timesteps, features)
    # The timesteps here will be the flattened spatial dimensions of the conv output
    x = Reshape((-1, cnn_output_shape[-1]))(x)
    
    # Add an LSTM layer
    x = LSTM(64)(x)
    
    # Dense layers for classification
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(img_input, output, name='cnn_lstm_classifier')
    return model

# Initialize and compile the CNN-LSTM model
model = build_cnn_lstm_model((HEIGHT, WIDTH, CHANNEL), NUM_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=EPOCHS,
    verbose=VERBOSE
)

# Evaluate the model
y_pred_proba = model.predict(X_test, verbose=VERBOSE)
y_pred = np.argmax(y_pred_proba, axis=1)

# Report on the model's performance
accuracy = accuracy_score(y_test, y_pred)
target_names = sorted(os.listdir(train_dir))
report = classification_report(y_test, y_pred, target_names=target_names)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the trained model
model.save("cnn_lstm_model.h5")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()

# Plotting training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_plot.png")
plt.show()
