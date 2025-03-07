import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from .ocsvm import OneClassSVM_QP

# Function to extract HOG features
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image).flatten()

# Load only "good" images for training
def load_good_data(folder):
    images = []
    path = folder
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))  # Resize for consistency
        features = extract_hog_features(image)  # Extract HOG features
        images.append(features)
    return np.array(images)

# Train and save One-Class SVM
def svm(train_folder="dataset_folder", model_path="oc_svm_model.pkl", scaler_path="scaler.pkl", retrain=False):
    if not retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model...")
        oc_svm = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Training new model...")
        X_good = load_good_data(train_folder)

        # Normalize features
        scaler = StandardScaler()
        X_good_scaled = scaler.fit_transform(X_good)

        # Train One-Class SVM (QP version)
        oc_svm = OneClassSVM_QP(kernel="rbf", gamma=0.1, nu=0.05)
        oc_svm.fit(X_good_scaled)

        # Save the trained model and scaler
        joblib.dump(oc_svm, model_path)
        joblib.dump(scaler, scaler_path)
        print("Model saved!")

    return oc_svm, scaler

# Function to test new images
def test_image(img_path, model_path="oc_svm_model.pkl", scaler_path="scaler.pkl"):
    oc_svm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    features = extract_hog_features(image).reshape(1, -1)
    features = scaler.transform(features)  # Apply saved normalization

    prediction = oc_svm.predict(features)
    
    return "Good" if prediction == 1 else "Bad (Anomaly)"

# Example usage
# svm()  # Train if not already trained
# print(test_image("dataset_folder/bad/sample1.jpg"))  # Test new image
