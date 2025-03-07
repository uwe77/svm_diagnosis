import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from .ocsvm import OneClassSVM_QP
from .knn import KNN


# ---------------------------------------
# HOG Feature Extraction (FIXED)
# ---------------------------------------
def extract_hog_features(image):
    """Extracts HOG features from an image."""
    hog = cv2.HOGDescriptor()
    return hog.compute(image).flatten()

# ---------------------------------------
# KNN-Based Image Segmentation (White & Black Classification)
# ---------------------------------------
class KNNImageSegmenter:
    """Uses NumPy KNN to classify each pixel into White (1) or Black (0)."""

    def __init__(self, k=5):
        self.k = k
        self.knn = KNN(k=self.k)

    def fit(self, image):
        """Train KNN to classify pixels as black or white based on intensity."""
        pixels = image.reshape(-1, 1)
        labels = (pixels > np.mean(pixels)).astype(np.uint8).flatten()

        self.knn.fit(pixels, labels)

    def transform(self, image):
        """Classify each pixel as black (0) or white (1) using KNN."""
        pixels = image.reshape(-1, 1)
        labels = self.knn.predict(pixels)
        return labels.reshape(image.shape)

# ---------------------------------------
# Compute Mean and Std (Manual Scaling)
# ---------------------------------------
def compute_mean_std(X):
    """Compute mean and std for manual normalization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # Avoid divide-by-zero
    return mean, std

def normalize_features(X, mean, std):
    """Normalize features using precomputed mean and std."""
    return (X - mean) / std

# ---------------------------------------
# Load "Good" Images and Apply KNN Segmentation
# ---------------------------------------
def load_good_data(folder):
    assert os.path.exists(folder), f"❌ Error: Folder not found! ({folder})"

    images = []
    valid_extensions = {".jpg", ".png"}

    for file in os.listdir(folder):
        if not any(file.lower().endswith(ext) for ext in valid_extensions):
            print(f"⚠️ Skipping non-image file: {file}")
            continue

        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"❌ Warning: Could not load image {img_path}. Skipping...")
            continue

        image = cv2.resize(image, (128, 128))

        knn_segmenter = KNNImageSegmenter(k=3)
        knn_segmenter.fit(image)
        segmented_image = knn_segmenter.transform(image)

        features = extract_hog_features(segmented_image)
        images.append(features)

    if len(images) == 0:
        raise FileNotFoundError(f"❌ No valid JPG/PNG images found in {folder}.")

    return np.array(images), knn_segmenter

# ---------------------------------------
# Train and Save One-Class SVM with KNN Segmentation
# ---------------------------------------
def svm(train_folder="dataset_folder", model_path="oc_svm_model.pkl", retrain=False):
    """
    Trains a One-Class SVM (QP) on "good" images.
    Uses NumPy-based KNN segmentation before feature extraction.
    Saves (oc_svm, mean, std, knn_segmenter) to 'model_path'.
    """
    if not retrain and os.path.exists(model_path):
        print("Loading existing model...")
        oc_svm, mean, std, knn_segmenter = joblib.load(model_path)
    else:
        print("Training new model...")
        X_good, knn_segmenter = load_good_data(train_folder)  # Get KNN model

        # Compute mean & std
        mean, std = compute_mean_std(X_good)

        # Normalize features
        X_good_normalized = normalize_features(X_good, mean, std)

        # Train One-Class SVM
        oc_svm = OneClassSVM_QP(kernel="rbf", gamma=0.1, nu=0.05)
        oc_svm.fit(X_good_normalized)

        # Save (oc_svm, mean, std, knn_segmenter) in a single file
        joblib.dump((oc_svm, mean, std, knn_segmenter), model_path)
        print("✅ Model trained and saved!")

    return oc_svm, mean, std, knn_segmenter  # RETURN ALL FOUR VALUES


# ---------------------------------------
# Test a New Image with KNN Segmentation
# ---------------------------------------
def test_image(img_path, model_path="oc_svm_model.pkl", plot=True):
    """
    Loads the trained model (oc_svm, mean, std, knn_segmenter) from 'model_path'.
    Applies NumPy KNN segmentation before HOG feature extraction.
    Predicts "Good" or "Bad (Anomaly)".
    """
    oc_svm, mean, std, knn_segmenter = joblib.load(model_path)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"❌ Error: Could not load image {img_path}")

    image = cv2.resize(image, (128, 128))

    segmented_image = knn_segmenter.transform(image)

    features = extract_hog_features(segmented_image).reshape(1, -1)
    features = normalize_features(features, mean, std)

    prediction = oc_svm.predict(features)
    result_label = "Good" if prediction == 1 else "Bad (Anomaly)"

    # ---------------------------------------
    # PLOTTING FUNCTION
    # ---------------------------------------
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(segmented_image, cmap="gray")
        ax[1].set_title(f"KNN Segmentation (Predicted: {result_label})")
        ax[1].axis("off")

        plt.show()

    return result_label

# ---------------------------------------
# Example Usage (Uncomment to run)
# ---------------------------------------
# if __name__ == "__main__":
#     svm(train_folder="dataset_folder/good", model_path="oc_svm_model.pkl", retrain=True)
#     test_image("dataset_folder/bad/sample1.jpg", model_path="oc_svm_model.pkl", plot=True)
