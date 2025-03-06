from setuptools import setup, find_packages

setup(
    name="svm-diagnosis",
    version="0.1.0",
    author="Yu Wei. Chang",
    author_email="uwe90711@gmail.com",
    description="A fault detection system using One-Class SVM for image classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uwe77/svm_diagnosis.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
