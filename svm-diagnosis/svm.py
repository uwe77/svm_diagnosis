import os
import cv2
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

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

# Train and save One-Class SVM
def svm(train_folder="dataset_folder", model_path="oc_svm_model.pkl", scaler_path="scaler.pkl", retrain=False):
    # Check if model exists and load it if retrain=False
    if not retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model...")
        oc_svm = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Training new model...")
        # Load training data (only "good" images)
        X_good = load_good_data(train_folder)

        # Normalize features
        scaler = StandardScaler()
        X_good_scaled = scaler.fit_transform(X_good)

        # Train One-Class SVM
        oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # nu controls sensitivity
        oc_svm.fit(X_good_scaled)

        # Save the trained model and scaler
        joblib.dump(oc_svm, model_path)
        joblib.dump(scaler, scaler_path)
        print("Model saved!")

    return oc_svm, scaler

# Function to test new images
def test_image(img_path, model_path="oc_svm_model.pkl", scaler_path="scaler.pkl"):
    # Load the trained model and scaler
    oc_svm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load and process the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    features = extract_hog_features(image).reshape(1, -1)
    features = scaler.transform(features)  # Apply saved normalization

    # Predict using SVM
    prediction = oc_svm.predict(features)
    
    return "Good" if prediction == 1 else "Bad (Anomaly)"

# Example Usage:
# Train the model (Only required the first time)
svm()

# Test an image
print(test_image("dataset_folder/bad/sample1.jpg"))  # Change to your test image path
