# svm_diagnosis

`svm_diagnosis` is a Python package for **fault detection in images using One-Class SVM**. It trains an **SVM model** to recognize "good" images and classify unknown samples as "good" or "bad (anomaly)".

## 🚀 Features
- **One-Class SVM-based anomaly detection**
- **HOG feature extraction**
- **Supports training, saving, and loading models**
- **Works with small datasets**
- **Uses OpenCV for image processing**

---

## 📥 Installation
### **Using pip (After Building Locally)**
```bash
pip install .
```

### **From Source**
```bash
git clone https://github.com/uwe77/svm_diagnosis.git
cd svm_diagnosis
pip install .
```

---

## 📌 Usage
### **1. Import the Package**
```python
import svm_diagnosis
```

### **2. Train the Model**
```python
from svm_diagnosis import svm

# Train and save the SVM model
svm(train_folder="dataset_folder", retrain=True)  # Set retrain=True to force model retraining
```

### **3. Test an Image**
```python
from svm_diagnosis import test_image

# Test a new image
result = test_image("dataset_folder/bad/sample1.jpg")
print("Diagnosis:", result)
```

### **4. Running in Docker**
If you're using Docker:
```bash
docker build -t svm-detector .
docker run --rm svm-detector
```

---

## 📂 Folder Structure
```
svm_diagnosis/
│── svm_diagnosis/
│   │── __init__.py
│   │── svm_model.py
│── tests/
│── setup.py
│── pyproject.toml
│── README.md
│── Dockerfile
│── dataset_folder/
│   │── good/  (Training images)
│   │── bad/   (Test images)
```

---

## ⚙️ Functions
### **`svm()` - Train and Save the SVM Model**
```python
svm(train_folder="dataset_folder", model_path="oc_svm_model.pkl", scaler_path="scaler.pkl", retrain=False)
```
- **`train_folder`**: Path to dataset (`good` images only)
- **`model_path`**: Where to save the SVM model
- **`scaler_path`**: Where to save the feature scaler
- **`retrain`**: Set `True` to retrain, `False` to load existing model

---

### **`test_image()` - Test a New Image**
```python
result = test_image("path/to/image.jpg")
```
- Returns `"Good"` if the image matches trained patterns.
- Returns `"Bad (Anomaly)"` if the image is outside the learned distribution.

---

## 🛠 Dependencies
- `numpy`
- `opencv-python`
- `scikit-learn`
- `joblib`

---

## 🤝 Contributing
Feel free to **open an issue** or submit a **pull request** if you want to improve this project! 🚀

---

## 📬 Contact
For questions, contact **Yu Wei. Chang** at [uwe90711@gmail.com](mailto:uwe90711@gmail.com).
```

---

### **What’s Improved?**
✅ **Clear step-by-step guide** for installation & usage  
✅ **Example code snippets** for training & testing  
✅ **Function documentation** for `svm()` and `test_image()`  
✅ **Docker usage included**  
✅ **Folder structure explanation**  