import os
import cv2
import numpy as np
import joblib
from .ocsvm import OneClassSVM_QP

# ---------------------------------------
# HOG Feature Extraction
# ---------------------------------------
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image).flatten()

# ---------------------------------------
# Compute Mean and Std (Manual Scaling)
# ---------------------------------------
def compute_mean_std(X):
    """Compute mean and std for manual normalization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # Add epsilon to avoid divide-by-zero
    return mean, std

def normalize_features(X, mean, std):
    """Normalize features using precomputed mean and std."""
    return (X - mean) / std

# ---------------------------------------
# Load "Good" Images
# ---------------------------------------
def load_good_data(folder):
    assert os.path.exists(folder), f"❌ Error: Folder not found! ({folder})"

    images = []
    valid_extensions = {".jpg", ".png"}  # Accept only JPG and PNG

    for file in os.listdir(folder):
        # Skip non-image files
        if not any(file.lower().endswith(ext) for ext in valid_extensions):
            print(f"⚠️ Skipping non-image file: {file}")
            continue

        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Skip if image can't be loaded
        if image is None:
            print(f"❌ Warning: Could not load image {img_path}. Skipping...")
            continue

        # Resize to a consistent dimension
        image = cv2.resize(image, (128, 128))

        # Extract HOG features
        features = extract_hog_features(image)
        images.append(features)

    if len(images) == 0:
        raise FileNotFoundError(f"❌ No valid JPG/PNG images found in {folder}.")

    return np.array(images)

# ---------------------------------------
# Train and Save One-Class SVM (No scikit-learn)
# ---------------------------------------
def svm(train_folder="dataset_folder", model_path="oc_svm_model.pkl", retrain=False):
    """
    Trains a One-Class SVM (QP) on "good" images.
    Saves (model, mean, std) to 'model_path'.
    If retrain=False and model_path exists, loads existing model.
    """
    if not retrain and os.path.exists(model_path):
        print("Loading existing model...")
        oc_svm, mean, std = joblib.load(model_path)
    else:
        print("Training new model...")
        X_good = load_good_data(train_folder)

        # Compute manual mean & std
        mean, std = compute_mean_std(X_good)

        # Normalize training data
        X_good_normalized = normalize_features(X_good, mean, std)

        # Train One-Class SVM (QP version)
        oc_svm = OneClassSVM_QP(kernel="rbf", gamma=0.1, nu=0.05)
        oc_svm.fit(X_good_normalized)

        # Save the model, mean, std together (NO scaler.pkl needed)
        joblib.dump((oc_svm, mean, std), model_path)
        print("Model trained and saved!")

    return oc_svm, mean, std

# ---------------------------------------
# Test a New Image
# ---------------------------------------
def test_image(img_path, model_path="oc_svm_model.pkl"):
    """
    Loads the trained model (oc_svm, mean, std) from 'model_path'.
    Normalizes the input image features using the same mean, std.
    Predicts "Good" or "Bad (Anomaly)".
    """
    # Load model + normalization params
    oc_svm, mean, std = joblib.load(model_path)

    # Read and preprocess image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"❌ Error: Could not load image {img_path}")

    image = cv2.resize(image, (128, 128))
    features = extract_hog_features(image).reshape(1, -1)

    # Normalize using training mean & std
    features = normalize_features(features, mean, std)

    # Predict
    prediction = oc_svm.predict(features)
    return "Good" if prediction == 1 else "Bad (Anomaly)"

# ---------------------------------------
# Example Usage (Uncomment to run)
# ---------------------------------------
# if __name__ == "__main__":
#     svm(train_folder="dataset_folder/good", model_path="oc_svm_model.pkl", retrain=True)
#     result = test_image("dataset_folder/bad/sample1.jpg", model_path="oc_svm_model.pkl")
#     print("Diagnosis:", result)
