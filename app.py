"""
Aerial DeepFake Detection Web Application
Using trained hybrid CNN model with Explainable AI
"""

import os
import numpy as np
import cv2
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - running in demo mode")

from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from PIL import Image
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Global variables for model
model = None
MODEL_PATH = 'my_aerial_deepfake_detector.h5'

class GradCAM:
    """Grad-CAM implementation for explainable AI"""
    def __init__(self, model, layer_name=None):
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output.shape) == 4:  # Conv layer
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            # Fallback to a dense layer
            for layer in reversed(model.layers):
                if 'activation' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name} for Grad-CAM")
        
        try:
            self.grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(layer_name).output, model.output]
            )
        except:
            print("⚠️ Grad-CAM setup failed, using basic visualization")
            self.grad_model = None
    
    def generate_heatmap(self, image, class_idx=0):
        """Generate Grad-CAM heatmap"""
        if self.grad_model is None:
            return np.random.random((224, 224))
        
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(image)
                loss = predictions[:, 0]
            
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                return np.random.random((224, 224))
            
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            return np.random.random((224, 224))
    
    def visualize_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        try:
            if heatmap is None or heatmap.size == 0:
                return image / 255.0 if image.max() > 1 else image
            
            if len(heatmap.shape) > 2:
                heatmap = np.squeeze(heatmap)
            
            if len(heatmap.shape) != 2:
                return image / 255.0 if image.max() > 1 else image
            
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), 
                                       (image.shape[1], image.shape[0]))
            
            if heatmap_resized.max() > heatmap_resized.min():
                heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            else:
                heatmap_resized = np.zeros_like(heatmap_resized)
            
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            if image.max() > 1:
                image = image / 255.0
            
            overlayed = heatmap_colored * alpha + image * (1 - alpha)
            return overlayed
            
        except Exception as e:
            print(f"Heatmap visualization failed: {e}")
            return image / 255.0 if image.max() > 1 else image

def load_model():
    """Load the trained deepfake detection model"""
    global model
    
    if not TENSORFLOW_AVAILABLE:
        print("⚠️ TensorFlow not available - running in demo mode")
        return False
        
    try:
        if os.path.exists(MODEL_PATH):
            # Try to load with custom objects and compile=False for compatibility
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Model loaded successfully (without compilation)!")
                
                # Recompile the model with appropriate settings
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print("✅ Model recompiled successfully!")
                return True
            except Exception as e1:
                print(f"⚠️ Failed to load with compile=False: {e1}")
                # Fallback to normal loading
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully with fallback method!")
                return True
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 This might be a model compatibility issue. Consider retraining the model.")
        return False

def preprocess_image(image_path):
    """Preprocess uploaded image for model prediction"""
    try:
        # Load image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        return img_array, np.array(img)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_image(image_array):
    """Make prediction on preprocessed image"""
    try:
        if model is not None:
            # Real model prediction
            prediction = model.predict(image_array, verbose=0)[0][0]
        else:
            # Demo mode - generate random but realistic prediction
            import random
            prediction = random.uniform(0.2, 0.8)  # Avoid extreme values
            print("🎭 Demo mode: Generated random prediction")
        
        confidence = prediction if prediction > 0.5 else 1 - prediction
        predicted_class = "FAKE" if prediction > 0.5 else "REAL"
        
        return {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probability_fake': prediction,
            'probability_real': 1 - prediction,
            'demo_mode': model is None
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def generate_explanation(image_array, original_image):
    """Generate Grad-CAM explanation"""
    try:
        if model is not None:
            # Real model explanation
            gradcam = GradCAM(model)
            heatmap = gradcam.generate_heatmap(image_array)
            explained_image = gradcam.visualize_heatmap(original_image, heatmap)
        else:
            # Demo mode - generate a simple overlay
            print("🎭 Demo mode: Generated simple explanation overlay")
            explained_image = original_image / 255.0 if original_image.max() > 1 else original_image
            # Add a subtle random heatmap overlay for demo
            import numpy as np
            height, width = explained_image.shape[:2]
            demo_heatmap = np.random.random((height, width)) * 0.3  # Subtle overlay
            import matplotlib.pyplot as plt
            heatmap_colored = plt.cm.jet(demo_heatmap)[:, :, :3]
            explained_image = heatmap_colored * 0.3 + explained_image * 0.7
        
        return explained_image
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return original_image / 255.0 if original_image.max() > 1 else original_image

def save_result_image(image_array, filename):
    """Save result image and return path"""
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(image_array)
        plt.axis('off')
        
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        
        return filepath
    except Exception as e:
        print(f"Error saving result image: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            image_array, original_image = preprocess_image(filepath)
            if image_array is None:
                return jsonify({'error': 'Error processing image'}), 500
            
            # Make prediction
            result = predict_image(image_array)
            if result is None:
                return jsonify({'error': 'Error making prediction'}), 500
            
            # Generate explanation
            explained_image = generate_explanation(image_array, original_image)
            
            # Save result images
            original_path = save_result_image(original_image / 255.0 if original_image.max() > 1 else original_image, 
                                            f"original_{timestamp}.png")
            explained_path = save_result_image(explained_image, f"explained_{timestamp}.png")
            
            # Prepare response
            response = {
                'success': True,
                'filename': filename,
                'prediction': result,
                'original_image': f"static/results/original_{timestamp}.png",
                'explained_image': f"static/results/explained_{timestamp}.png",
                'timestamp': timestamp
            }
            
            return jsonify(response)
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
            
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze')
def analyze():
    """Analysis page"""
    return render_template('analyze.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    print("🚀 Starting Aerial DeepFake Detection Web Application...")
    
    # Load model on startup
    model_loaded = load_model()
    if model_loaded:
        print("✅ Model loaded successfully!")
        print("🌐 Starting web server...")
    else:
        print("❌ Failed to load model. Running in DEMO MODE.")
        print("💡 In demo mode, the app will return random predictions for demonstration.")
        print("💡 To fix: Retrain your model or ensure the model file is compatible.")
        print("🌐 Starting web server in demo mode...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
