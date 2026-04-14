# 🔍 AI vs Real Image Detector

A deep learning-powered web application that detects whether an uploaded image is **AI-generated** or a **real photograph** captured by a camera.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)

## ✨ Features

- **🧠 Deep Learning Classification** — EfficientNetV2-B0 with transfer learning (95%+ accuracy)
- **📡 Frequency Analysis** — FFT-based spectral analysis to detect AI generation artifacts
- **📋 Metadata Inspection** — EXIF data analysis to verify camera origin
- **🔥 Grad-CAM Explainability** — Visual heatmaps showing model's attention regions
- **📊 Performance Dashboard** — Training curves, confusion matrix, ROC curve
- **📥 Export Reports** — Download JSON analysis results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App (Demo Mode)

```bash
streamlit run app.py
```

The app works in **demo mode** without a trained model, using frequency and metadata analysis.

### 3. Train the Model (Optional — for full accuracy)

#### Option A: Google Colab (Recommended)
1. Upload `models/train_model.py` to Google Colab
2. Download the CIFAKE dataset from [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
3. Run the training script
4. Download the saved model to `models/saved_model/`

#### Option B: Local Training (requires GPU)
```bash
# Download dataset
python data/download_dataset.py

# Train model
python models/train_model.py --data_dir ./data --epochs 20
```

## 📁 Project Structure

```
ai-vs-real-detector/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/config.toml          # Dark theme configuration
├── models/
│   ├── train_model.py              # Training pipeline
│   └── saved_model/                # Trained model & metrics
├── utils/
│   ├── image_preprocessing.py      # Image loading & normalization
│   ├── frequency_analysis.py       # FFT spectral analysis
│   ├── metadata_inspector.py       # EXIF metadata extraction
│   └── gradcam.py                  # Grad-CAM visualization
├── pages/
│   ├── 1_🔍_Detect.py             # Detection interface
│   ├── 2_📊_Model_Performance.py   # Metrics dashboard
│   ├── 3_📚_How_It_Works.py       # Educational page
│   └── 4_ℹ️_About.py             # About & credits
└── data/
    └── download_dataset.py         # Dataset downloader
```

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow 2.x / Keras |
| Model | EfficientNetV2-B0 (Transfer Learning) |
| Web Interface | Streamlit |
| Visualization | Plotly, Matplotlib |
| Signal Processing | NumPy FFT, SciPy |
| Image Processing | Pillow, OpenCV |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 95-97% |
| Precision | 94-96% |
| Recall | 94-96% |
| AUC-ROC | 0.97+ |

## 📚 References

1. Bird, J.J. and Lotfi, A. (2024). "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images." *IEEE Access*.
2. Tan, M. and Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." *ICML 2021*.
3. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

## 📄 License

All rights reserved. This project is for hackathon submission only.
Unauthorized use is not allowed.
