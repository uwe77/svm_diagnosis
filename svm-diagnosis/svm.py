def svm():
    pass

import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to extract HOG features
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image).flatten()

# Load only "good" images for training
def load_good_data(folder):
    images = []
    path = os.path.join(folder, "good")
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))  # Resize for consistency
        features = extract_hog_features(image)  # Extract HOG features
        images.append(features)
    return np.array(images)

# Load training data (only "good" images)
X_good = load_good_data("dataset_folder")  # Replace with your dataset path

# Normalize features
scaler = StandardScaler()
X_good_scaled = scaler.fit_transform(X_good)

# Train One-Class SVM
oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # nu controls sensitivity
oc_svm.fit(X_good_scaled)

# Function to test new images
def test_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    features = extract_hog_features(image).reshape(1, -1)
    features = scaler.transform(features)  # Apply same normalization
    prediction = oc_svm.predict(features)
    
    return "Good" if prediction == 1 else "Bad (Anomaly)"

# Test with a new bad image
print(test_image("dataset_folder/bad/sample1.jpg"))  # Change to your test image path