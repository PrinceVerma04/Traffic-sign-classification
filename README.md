

<img width="1146" height="325" alt="traffic sign" src="https://github.com/user-attachments/assets/516a6936-9d58-4779-8fd6-53de87b5ea29" />

# Traffic Sign Classification Using CNN

A deep learning project that implements a Convolutional Neural Network (CNN) to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset with **99.5% accuracy**.

## 🚦 Project Overview

This project demonstrates the power of CNNs in image classification by accurately recognizing 43 different types of traffic signs. The model achieves exceptional performance through a carefully designed architecture and training strategy.

# 📊 Dataset
## Option 1: Using kaggle CLI
kaggle datasets download meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Option 2: Manual download
## Visit: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
## Download and extract to data/raw/ directory

**German Traffic Sign Recognition Benchmark (GTSRB)**
- **43 classes** of traffic signs
- **39,209 training images**
- **7,842 testing images**
- **Image size**: 30×30 pixels (RGB)
- Classes include speed limits, stop signs, yield signs, and various warning signs

## 🏗️ Model Architecture

<img width="513" height="408" alt="model_architecture" src="https://github.com/user-attachments/assets/ddf31c5e-afdb-4f7e-a6ba-0790e0933817" />


The CNN model consists of:
- **4 Convolutional Layers** with increasing filters (32 → 64 → 128 → 256)
- **2 MaxPooling Layers** for dimensionality reduction
- **3 Dropout Layers** (15%, 20%, 25%) for regularization
- **2 Fully Connected Layers** (512 → 43 units)
- **Total Parameters**: 1,624,939

## 📈 Performance
<img width="1188" height="372" alt="model accuracy" src="https://github.com/user-attachments/assets/cc184b63-1f26-4e57-856e-3747628248f5" />

- **Training Accuracy**: 99.5%
- **Validation Accuracy**: 99.5%
- **Loss**: Minimal convergence
- **Training Time**: ~3 seconds per epoch on GPU

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- matplotlib, seaborn

### Install Dependencies
```bash
pip install tensorflow opencv-python pillow scikit-learn matplotlib seaborn pandas numpy
```
## Note: You can download the complete notebook from the given notebook directory
