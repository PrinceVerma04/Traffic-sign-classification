# Traffic Sign Classification Using CNN

A deep learning project that implements a Convolutional Neural Network (CNN) to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset with **99.5% accuracy**.

## 🚦 Project Overview

This project demonstrates the power of CNNs in image classification by accurately recognizing 43 different types of traffic signs. The model achieves exceptional performance through a carefully designed architecture and training strategy.

## 📊 Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- **43 classes** of traffic signs
- **39,209 training images**
- **7,842 testing images**
- **Image size**: 30×30 pixels (RGB)
- Classes include speed limits, stop signs, yield signs, and various warning signs

## 🏗️ Model Architecture

The CNN model consists of:
- **4 Convolutional Layers** with increasing filters (32 → 64 → 128 → 256)
- **2 MaxPooling Layers** for dimensionality reduction
- **3 Dropout Layers** (15%, 20%, 25%) for regularization
- **2 Fully Connected Layers** (512 → 43 units)
- **Total Parameters**: 1,624,939

## 📈 Performance

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
