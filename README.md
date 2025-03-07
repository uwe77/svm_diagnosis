# One-Class SVM for Anomaly Detection (Quadratic Programming)

`svm_diagnosis` is a Python package for **anomaly detection using One-Class SVM with Quadratic Programming (QP)**. This package helps classify images as either "Good" or "Bad (Anomaly)" by learning from only "Good" images.

## 🚀 Features
- **One-Class SVM (QP-based implementation)**
- **No need for labeled "bad" images** (unsupervised learning)
- **HOG (Histogram of Oriented Gradients) feature extraction**
- **Saves and loads trained models for reuse**
- **OpenCV-based image processing**
- **Supports RBF and Linear kernels**

---

## 📥 Installation
### **Install via pip (After Cloning the Repo)**
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
from svm_diagnosis import svm, test_image
```

### **2. Train the Model**
```python
# Train using "Good" images from the dataset folder
svm(train_folder="dataset_folder/good", retrain=True)
```

### **3. Test an Image**
```python
# Test a new image and print the classification result
result = test_image("dataset_folder/bad/sample1.jpg")
print("Diagnosis:", result)  # Expected output: "Bad (Anomaly)"
```

---

## 📂 Folder Structure
```
svm_diagnosis/
│── svm_diagnosis/
│   │── __init__.py
│   │── ocsvm.py  # One-Class SVM with QP
│   │── svm_model.py  # Training & Prediction Functions
│── tests/
│── setup.py
│── pyproject.toml
│── README.md
│── Docker/
│   │── cpu/
│   │── app_linux/
│   │── app_windows/
│── images/
│   ├── good/  (Training images)
│   ├── bad/   
```

---

## 📜 API Reference

### **Train the One-Class SVM**
```python
svm(train_folder="dataset_folder/good", model_path="oc_svm_model.pkl", retrain=False)
```
#### **Parameters**
- `train_folder` *(str)* – Path to the folder containing only "Good" images.
- `model_path` *(str)* – Path to save the trained model.
- `retrain` *(bool)* – If `True`, forces retraining even if a saved model exists.

#### **Returns**
- Trained `OneClassSVM_QP` model.

---

### **Test a New Image**
```python
result = test_image("dataset_folder/bad/sample1.jpg")
```
#### **Parameters**
- `img_path` *(str)* – Path to the image to classify.
- `model_path` *(str, default="oc_svm_model.pkl")* – Path to the saved model.

#### **Returns**
- `"Good"` if the image is classified as normal.
- `"Bad (Anomaly)"` if classified as an outlier.

---

## 🏗 Implementation Details

### **One-Class SVM (Quadratic Programming)**
This package uses a **custom QP-based One-Class SVM** for anomaly detection. The key equations are:

1. **Decision function:**  
   \[
   f(x) = \sum_{i} \alpha_i K(x_i, x) - \rho
   \]
   Where:
   - \( \alpha \) are Lagrange multipliers.
   - \( K(x_i, x) \) is the kernel function (RBF or Linear).
   - \( \rho \) is the decision boundary.

2. **Optimization Problem (QP Formulation):**
   \[
   \min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j K(x_i, x_j) - \sum_i \alpha_i
   \]
   Subject to:
   - \( 0 \leq \alpha_i \leq \frac{1}{\nu n} \)
   - \( \sum_i \alpha_i = 1 \)

---

## 🛠 Troubleshooting & FAQ

### **Q: My "Good" images are misclassified as anomalies. What should I do?**
✔ Ensure the **training and testing images** are processed with the same normalization.  
✔ Increase `nu` (e.g., `nu=0.1`) to allow more variance in good samples.  
✔ Decrease `gamma` (e.g., `gamma=0.01`) to make the decision boundary less strict.

### **Q: I get a `cv2.error: (-215:Assertion failed) !ssize.empty() in function 'resize'`.**
✔ Check if the **image path is correct** before passing it to `cv2.imread()`.  
✔ Add this check before resizing:
```python
if image is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")
```

### **Q: How can I change the kernel from RBF to Linear?**
✔ Modify `svm_model.py`:
```python
oc_svm = OneClassSVM_QP(kernel="linear", gamma=0.1, nu=0.05)
```

---

## 🤝 Contributing
Feel free to **open an issue** or submit a **pull request** if you want to improve this project! 🚀

---

## 📬 Contact
For questions, contact **Yu Wei. Chang** at [uwe90711@gmail.com](mailto:uwe90711@gmail.com).

---

## **🔹 What’s New?**
✅ **Updated to match current code (One-Class SVM with QP)**  
✅ **Added API documentation with function descriptions**  
✅ **Provided FAQ & Troubleshooting tips**  
✅ **Explained Quadratic Programming equations for One-Class SVM**  

---

### **🛠 Next Steps**
1️⃣ **Verify folder structure**: Ensure images are inside `dataset_folder/good` and `dataset_folder/bad`.  
2️⃣ **Run tests**:
   ```bash
   python test_svm.py
   ```
3️⃣ **Test image classification manually**:
   ```python
   from svm_diagnosis import test_image
   print(test_image("dataset_folder/bad/sample1.jpg"))  # Should output: "Bad (Anomaly)"
   ```

---
