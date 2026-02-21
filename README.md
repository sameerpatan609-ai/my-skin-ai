# SkinCare AI: End-to-End Dermatology Assistant
# Skin Care AI - Diagnostic Analysis System

### 🌐 Permanent Live Link:
**[https://my-skin-ai-yfyw.onrender.com](https://my-skin-ai-yfyw.onrender.com)**

This project detects skin conditions using a trained MobileNetV2 model and provides personalized product recommendations based on a diagnostic report.sion

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=YOUR_GITHUB_REPO_URL)

SkinCare AI is a production-quality machine learning application that detects facial skin conditions (Acne, Dark Spots, Uneven Texture) from images or live camera feeds and recommends optimized skincare products.

## 🚀 Key Features
- **Computer Vision**: Transfer learning with MobileNetV2 trained on dermatology-specific patterns.
- **Smart Prprocessing**: Automatic face detection and ROI cropping using Haar Cascades.
- **Hotspot Visualization**: Highlights detected anomalies directly on the facial region.
- **Recommendation Engine**: Content-based filtering using TF-IDF to map conditions to suitable ingredients.
- **Premium UI**: Modern glassmorphism design with responsive layout and live camera support.

## 🛠️ Tech Stack
- **Backend**: Python (Flask)
- **ML/CV**: TensorFlow, OpenCV, Scikit-Learn
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (ES6+)

## 📦 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python utils/generate_data.py
```

### 3. Train Model
```bash
python train_model.py
```
*Note: The app will automatically fallback to a `MockPredictor` if the model is not found.*

### 4. Run Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## 🧪 Testing
Run the automated test suite:
```bash
python -m unittest tests/test_app.py
```

## 📂 Project Structure
- `app.py`: Main Flask application entry point.
- `train_model.py`: CNN training pipeline.
- `recommendation.py`: Product mapping engine.
- `utils/visualizer.py`: Hotspot detection logic.
- `templates/index.html`: Premium UI.
- `data/products.csv`: Curated skincare database.
