from setuptools import setup, find_packages

setup(
    name="svm_diagnosis",
    version="0.1.0",
    author="Yu Wei. Chang",
    author_email="uwe90711@gmail.com",
    description="A fault detection system using One-Class SVM with QP for image classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uwe77/svm_diagnosis.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "joblib",
        "cvxopt"  # Added cvxopt for QP solver
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
