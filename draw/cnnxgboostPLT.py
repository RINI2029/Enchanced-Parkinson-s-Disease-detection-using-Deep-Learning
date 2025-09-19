import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set up directories and parameters
base_dir = r"C:\Users\Administrator\Desktop\parkino\draw\dataset/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "val")
WIDTH, HEIGHT, CHANNEL = 224, 224, 3
NUM_CLASSES = 0  # To be determined dynamically
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

# Build the CNN model for feature extraction
def build_cnn_model(input_shape):
    img_input = Input(shape=input_shape)
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
    x = Flatten()(x)
    model = Model(img_input, x, name='cnn_feature_extractor')
    return model

# Initialize and compile the CNN model
feature_extractor = build_cnn_model((HEIGHT, WIDTH, CHANNEL))
feature_extractor.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
feature_extractor.summary()

# Save the CNN feature extractor model
feature_extractor.save("cnn_feature_extractor.h5")

# Extract features using the CNN model
X_train_features = feature_extractor.predict(X_train, verbose=VERBOSE)
X_test_features = feature_extractor.predict(X_test, verbose=VERBOSE)

print(f"Shape of extracted features in X_train_features: {X_train_features.shape}")
print(f"Shape of extracted features in X_test_features: {X_test_features.shape}")

# Train XGBoost classifier on the extracted features
xgb_classifier = XGBClassifier(objective='multi:softprob', num_class=NUM_CLASSES)
xgb_classifier.fit(X_train_features, y_train)

# Predict on test set and get class labels
y_pred_proba = xgb_classifier.predict_proba(X_test_features)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
# Correctly get target names
target_names = sorted(os.listdir(train_dir))
report = classification_report(y_test, y_pred, target_names=target_names)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the XGBoost model
xgb_classifier.save_model("xgb_classifier_model.json")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()

# Feature Importance
importance = xgb_classifier.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.savefig("feature_importance.png")
plt.show()

# ROC Curve and AUC
y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)
plt.figure(figsize=(12, 8))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve.png")
plt.show()
