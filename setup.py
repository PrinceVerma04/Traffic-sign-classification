from setuptools import setup, find_packages

setup(
    name="traffic-sign-classification",
    version="1.0.0",
    description="Traffic Sign Classification using CNN",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.7",
)
