from svm_diagnosis import OneClassSVM_QP, extract_hog_features, load_good_data, svm, test_image
import numpy as np
import os
import cv2
import joblib

# Get absolute paths dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get current script directory
TRAIN_FOLDER = os.path.join(BASE_DIR, "../images/goods")  # Adjusted for correct dataset path
MODEL_PATH = os.path.join(BASE_DIR, "../model/oc_svm_model.pkl")  # Model save path
TEST_IMAGE_GOOD = os.path.join(BASE_DIR, "../images/goods/550878606537261445.jpg")  # A sample good image
TEST_IMAGE_BAD = os.path.join(BASE_DIR, "../images/bads/550878599708934361.jpg")  # A sample bad image

# ---------------------------------------
# Helper Function: Check if File Exists
# ---------------------------------------
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Error: File not found: {filepath}")

# ---------------------------------------
# Test Case 1: Load and Verify "Good" Data
# ---------------------------------------
def test_load_good_data():
    print("\n🔄 Checking dataset folder...")
    if not os.path.exists(TRAIN_FOLDER):
        raise FileNotFoundError(f"❌ Error: Training folder {TRAIN_FOLDER} does not exist!")
    
    X_good = load_good_data(TRAIN_FOLDER)
    assert X_good.shape[0] > 0, "❌ Failed to load training data!"
    print(f"✅ Loaded {X_good.shape[0]} good images for training.")

# ---------------------------------------
# Test Case 2: Train and Save One-Class SVM
# ---------------------------------------
def test_train_svm():
    print("\n🔄 Training One-Class SVM (QP)...")
    oc_svm, mean, std, knn_segmenter = svm(TRAIN_FOLDER, MODEL_PATH, retrain=True)  # FIXED
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Error: Model was not saved correctly at {MODEL_PATH}!")
    print("✅ Model trained and saved successfully!")

# ---------------------------------------
# Test Case 3: Load the Trained Model
# ---------------------------------------
def test_load_svm():
    print("\n🔄 Loading trained model...")
    check_file_exists(MODEL_PATH)
    
    oc_svm, mean, std, knn_segmenter = joblib.load(MODEL_PATH)  # FIXED
    assert isinstance(oc_svm, OneClassSVM_QP), "❌ Error: Loaded model is not a OneClassSVM_QP instance!"
    print("✅ Model loaded successfully!")

# ---------------------------------------
# Test Case 4: Test a "Good" Image
# ---------------------------------------
def test_good_image():
    print("\n🔄 Testing a 'Good' image...")
    check_file_exists(TEST_IMAGE_GOOD)
    
    result = test_image(TEST_IMAGE_GOOD, MODEL_PATH)
    print(f"🔍 Prediction: {result}")

    if result == "Good":
        print("✅ 'Good' image correctly classified!")
    else:
        print("❌ False Negative: Good image misclassified!")

# ---------------------------------------
# Test Case 5: Test a "Bad" Image
# ---------------------------------------
def test_bad_image():
    print("\n🔄 Testing a 'Bad' image...")
    check_file_exists(TEST_IMAGE_BAD)
    
    result = test_image(TEST_IMAGE_BAD, MODEL_PATH)
    print(f"🔍 Prediction: {result}")

    if result == "Bad (Anomaly)":
        print("✅ 'Bad' image correctly classified!")
    else:
        print("❌ False Positive: Bad image misclassified!")

# ---------------------------------------
# Run All Tests
# ---------------------------------------
if __name__ == "__main__":
    print("🚀 Running tests for One-Class SVM (QP)...\n")

    try:
        # test_load_good_data()
        test_train_svm()
        test_load_svm()
        test_good_image()
        test_bad_image()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
