import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from recommendation import Recommender
from utils.visualizer import detect_hotspots
import time

# Constants
CLASS_NAMES = ["Acne", "Dark Spots", "Normal", "Uneven Texture"]

# OpenCV Face Detection Setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Attempt to load trained model, else mock
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    
    MODEL_PATH = "models/skin_model.h5"
    if os.path.exists(MODEL_PATH):
        print("Loading trained model...")
        model = load_model(MODEL_PATH)
        USE_MOCK = False
    else:
        print("Model file not found. Using MockPredictor.")
        from mock_predictor import MockPredictor
        model = MockPredictor()
        USE_MOCK = True

except (ImportError, Exception) as e:
    print(f"TensorFlow not available or error: {e}. Using MockPredictor.")
    from mock_predictor import MockPredictor
    model = MockPredictor()
    USE_MOCK = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Recommender
recommender = Recommender("data/products.csv")

def detect_and_crop_face(img_path):
    """
    Detects faces in an image. If Haar fails, falls back to skin-color presence check.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, False
    
    # Check if cascade loaded
    if face_cascade.empty():
        print("ERROR: Face cascade record is empty. Falling back to skin-color detection.")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            padding = 0.2
            xmin, ymin = max(0, int(x - padding * w)), max(0, int(y - padding * h))
            xmax, ymax = min(img.shape[1], int(x + w * (1 + padding))), min(img.shape[0], int(y + h * (1 + padding)))
            return img[ymin:ymax, xmin:xmax], True

    # FALLBACK: Skin color detection
    # Convert to HSV and check for high presence of skin tones
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
    
    print(f"Skin area ratio detected: {skin_ratio:.4f}")
    
    if skin_ratio > 0.15: # If more than 15% is skin-colored
        return img, True
    
    return img, False

def prepare_image(img_array):
    if USE_MOCK:
        return img_array # Not used by mock but keeping structure
    
    # Resizing for MobileNetV2
    img = cv2.resize(img_array, (224, 224))
    x = img.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Face Detection and Preprocessing
        face_img, face_detected = detect_and_crop_face(filepath)
        
        if not face_detected:
            # If no face is detected, we don't proceed with analysis
            os.remove(filepath) # Optional: clean up the invalid file
            return jsonify({'error': 'No skin/face detected. Please upload a clear photo of your face.'}), 400
        
        # Save processed/cropped face for frontend display
        # First, detect hotspots based on prediction if needed later, 
        # but let's do it after we get the predicted_class
        processed_filename = "proc_" + filename
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        # We'll overwrite this after prediction with hotspots
        
        # Inference
        try:
            if USE_MOCK:
                time.sleep(1)
                predicted_class, confidence = model.predict(filepath)
                probabilities = {c: 0.1 for c in CLASS_NAMES}
                probabilities[predicted_class] = confidence
                remainder = 1.0 - confidence
                other_classes = [c for c in CLASS_NAMES if c != predicted_class]
                for c in other_classes:
                    probabilities[c] = round(remainder / len(other_classes), 2)
            else:
                img_tensor = prepare_image(face_img)
                preds = model.predict(img_tensor)
                class_idx = np.argmax(preds[0])
                predicted_class = CLASS_NAMES[class_idx]
                confidence = round(float(preds[0][class_idx]), 2)
                probabilities = {class_name: round(float(score), 2) for class_name, score in zip(CLASS_NAMES, preds[0])}
            
            # Apply Hotspot Visualization
            visualized_img = detect_hotspots(face_img, predicted_class)
            cv2.imwrite(processed_path, visualized_img)
            
            # Get Recommendations
            recommendations = recommender.get_recommendations(predicted_class)
            
            return jsonify({
                'condition': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'recommendations': recommendations,
                'image_url': f"/static/uploads/{filename}",
                'face_url': f"/static/uploads/{processed_filename}",
                'face_detected': face_detected
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)

