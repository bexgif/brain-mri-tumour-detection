# 🧠 Brain MRI Tumour Detection

This project is a deep learning-based image classifier that detects the presence of brain tumours in MRI scans. Built using PyTorch and pretrained ResNet18, it includes a Google Colab notebook for model training and a Streamlit web app for real-time prediction.

## 🚀 Project Overview

- **Objective:** Classify brain MRI images as either `Tumour` or `No Tumour`
- **Frameworks Used:** PyTorch, torchvision, scikit-learn, Streamlit
- **Model:** ResNet18 (transfer learning)
- **Interface:** Upload an image to receive a prediction with confidence score

## 📂 Files Included

| File | Description |
|------|-------------|
| `brain_mri_classifier.ipynb` | Full training pipeline using Google Colab |
| `README.md` | Project overview |

## 🧠 How It Works

1. Images are preprocessed (resized, normalised, augmented).
2. A pretrained ResNet18 model is fine-tuned for binary classification.
3. The trained model predicts tumour presence with a confidence score.
4. Users can upload an image via the web interface to test the model.

## Dataset
Navoneel Chakrabarty- https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
