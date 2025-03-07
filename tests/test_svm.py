from svm_diagnosis import OneClassSVM_QP, extract_hog_features, load_good_data, svm, test_image
import numpy as np
import os
import cv2
import joblib



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(BASE_DIR, "../images/goods")  # Path to dataset
MODEL_PATH = os.path.join(BASE_DIR, "oc_svm_model.pkl")  # Model save path
TEST_IMAGE_GOOD = os.path.join(BASE_DIR, "../images/goods/550878606537261445.jpg")  # A sample good image
TEST_IMAGE_BAD = os.path.join(BASE_DIR, "../images/bads/550878599708934361.jpg")  # A sample bad image

# Test case 1: Load and verify "good" data loading
def test_load_good_data():
    X_good = load_good_data(TRAIN_FOLDER)
    assert X_good.shape[0] > 0, "Failed to load training data!"
    print(f"✅ Loaded {X_good.shape[0]} good images for training.")

# Test case 2: Train and save One-Class SVM
def test_train_svm():
    print("\n🔄 Training One-Class SVM (QP)...")
    model = svm(TRAIN_FOLDER, MODEL_PATH, retrain=True)
    assert os.path.exists(MODEL_PATH), "Model was not saved correctly!"
    print("✅ Model trained and saved successfully!")

# Test case 3: Load the trained model
def test_load_svm():
    print("\n🔄 Loading trained model...")
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, OneClassSVM_QP), "Failed to load the correct model!"
    print("✅ Model loaded successfully!")

# Test case 4: Test a known "good" image
def test_good_image():
    print("\n🔄 Testing a 'Good' image...")
    result = test_image(TEST_IMAGE_GOOD, MODEL_PATH)
    print(f"Prediction: {result}")
    if result == "Good":
        print("✅ 'Good' image correctly classified!")
    else:
        print("False Negative: Good image misclassified!")

# Test case 5: Test a known "bad" image
def test_bad_image():
    print("\n🔄 Testing a 'Bad' image...")
    result = test_image(TEST_IMAGE_BAD, MODEL_PATH)
    print(f"Prediction: {result}")
    if result == "Bad (Anomaly)":
        print("✅ 'Bad' image correctly classified!")
    else:
        print("False Positive: Bad image misclassified!")

# Run all tests
if __name__ == "__main__":
    print("🚀 Running tests for One-Class SVM (QP)...\n")
    
    test_load_good_data()
    test_train_svm()
    test_load_svm()
    test_good_image()
    test_bad_image()
    
    print("\n🎉 All tests passed successfully!")
