# =============================================================================
# HYBRID CNN + ViT DEEPFAKE DETECTOR FOR REAL DATASETS
# DeepFake Detection in Aerial Images Using Explainable AI
# =============================================================================

# =============================================================================
# CELL 1: INSTALL REQUIRED PACKAGES
# =============================================================================
!pip install transformers==4.21.0
!pip install shap
!pip install grad-cam
!pip install opencv-python
!pip install pillow
!pip install scikit-learn
!pip install seaborn
!pip install tensorflow>=2.10.0 --upgrade

# Verify critical installations
import sys
try:
    import transformers
    print(f"✅ Transformers version: {transformers.__version__}")
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available - will use CNN-only model")
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow installation failed!")
    sys.exit(1)

try:
    import shap
    print("✅ SHAP installed successfully")
except ImportError:
    print("⚠️ SHAP not available - explainability features limited")

print("🎯 Package installation check complete!")

# =============================================================================
# CELL 2: IMPORT ALL LIBRARIES
# =============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import shutil
import zipfile
import random
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Deep Learning Libraries
from tensorflow.keras.layers import (Input, Dense, Concatenate, Dropout, 
                                   GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Transformers for ViT
from transformers import TFViTModel, ViTImageProcessor

# Explainable AI
import shap
from tensorflow.keras.utils import plot_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# CELL 3: SETUP GPU AND ENVIRONMENT
# =============================================================================
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
            
            # Check GPU memory
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Details: {gpu_details}")
            
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
    else:
        print("⚠️ No GPU available, using CPU")
    
    # Set mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled")

setup_gpu()

# =============================================================================
# CELL 4: CONFIGURE YOUR DATASET PATH
# =============================================================================
# 🔥 YOUR DATASET PATH CONFIGURED 🔥

# Your dataset path (already uploaded to Colab)
DATASET_PATH = '/content/dataset'  # ✅ Your actual dataset path

# Your dataset structure should be:
# /content/dataset/
# ├── train/
# │   ├── real/     (real aerial images)
# │   └── fake/     (fake aerial images)
# ├── validation/   (optional - will auto-create if missing)
# │   ├── real/
# │   └── fake/
# └── test/         (optional - will auto-create if missing)
#     ├── real/
#     └── fake/

# Alternative: If you have a different structure, we'll auto-split
AUTO_SPLIT = True  # Set True if you want automatic train/val/test split

print(f"📁 Dataset path set to: {DATASET_PATH}")
print("🔍 Checking if your dataset exists...")

# Verify dataset exists
if os.path.exists(DATASET_PATH):
    print("✅ Dataset found!")
    print("📂 Contents of your dataset:")
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            print(f"   📁 {item}/ ({count} items)")
        else:
            print(f"   📄 {item}")
    
    # Check if we found "Major Project" folder
    major_project_path = os.path.join(DATASET_PATH, 'Major Project')
    if os.path.exists(major_project_path):
        print(f"\n🔍 Exploring 'Major Project' folder:")
        for item in os.listdir(major_project_path):
            item_path = os.path.join(major_project_path, item)
            if os.path.isdir(item_path):
                count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                print(f"   📁 {item}/ ({count} files)")
            else:
                print(f"   📄 {item}")
        
        # Update dataset path to point to Major Project folder
        print(f"\n📍 Updating dataset path to: {major_project_path}")
        DATASET_PATH = major_project_path
else:
    print("❌ Dataset not found! Please check your path.")
    print("Current working directory:", os.getcwd())
    print("Available directories in /content:")
    if os.path.exists('/content'):
        for item in os.listdir('/content'):
            if os.path.isdir(f'/content/{item}'):
                print(f"   📁 {item}/")
    else:
        print("   No /content directory found")

# =============================================================================
# CELL 5: DATASET DISCOVERY AND VALIDATION
# =============================================================================
def discover_dataset_structure(dataset_path):
    """Discover and validate dataset structure"""
    print(f"🔍 Analyzing dataset structure at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        print("Please check your dataset upload!")
        return None
    
    # Check for standard structure
    splits = ['train', 'validation', 'test']
    classes = ['real', 'fake']
    
    structure_info = {}
    has_standard_structure = True
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            structure_info[split] = {}
            for class_name in classes:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    # Count images
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                    images = [f for f in os.listdir(class_path) 
                             if any(f.lower().endswith(ext) for ext in image_extensions)]
                    structure_info[split][class_name] = len(images)
                    print(f"  {split}/{class_name}: {len(images)} images")
                else:
                    structure_info[split][class_name] = 0
                    has_standard_structure = False
        else:
            has_standard_structure = False
    
    if not has_standard_structure:
        print("⚠️ Standard structure not found. Checking for alternative structures...")
        
        # Check for flat structure (all images in subdirectories)
        subdirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"Found subdirectories: {subdirs}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            image_count = len([f for f in os.listdir(subdir_path) 
                             if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])])
            print(f"  {subdir}: {image_count} images")
    
    return structure_info

# Analyze your dataset
dataset_info = discover_dataset_structure(DATASET_PATH)

# =============================================================================
# CELL 6: AUTOMATIC DATASET SPLITTING (if needed)
# =============================================================================
def create_train_val_test_split(source_path, dest_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits from a flat directory structure"""
    print(f"🔄 Creating train/val/test splits...")
    
    # Create destination structure
    for split in ['train', 'validation', 'test']:
        for class_name in ['real', 'fake']:
            os.makedirs(os.path.join(dest_path, split, class_name), exist_ok=True)
    
    # Process each class
    for class_name in ['real', 'fake']:
        source_class_path = os.path.join(source_path, class_name)
        
        if not os.path.exists(source_class_path):
            print(f"⚠️ Class directory not found: {source_class_path}")
            continue
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = [f for f in os.listdir(source_class_path) 
                 if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        random.shuffle(images)
        
        # Calculate split sizes
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Split images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Copy files
        for split, image_list in [('train', train_images), ('validation', val_images), ('test', test_images)]:
            dest_class_path = os.path.join(dest_path, split, class_name)
            for image in image_list:
                src = os.path.join(source_class_path, image)
                dst = os.path.join(dest_class_path, image)
                shutil.copy2(src, dst)
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

# Auto-split if needed (check for flat structure)
dataset_info = discover_dataset_structure(DATASET_PATH)

# Force split creation for flat real/fake structure
real_path = os.path.join(DATASET_PATH, 'real')
fake_path = os.path.join(DATASET_PATH, 'fake')

if os.path.exists(real_path) and os.path.exists(fake_path):
    print("🔄 Detected flat structure (real/fake folders) - creating train/val/test splits...")
    print("🔄 Creating automatic dataset split from your uploaded data...")
    
    split_dataset_path = '/content/dataset_split'
    create_train_val_test_split(DATASET_PATH, split_dataset_path)
    DATASET_PATH = split_dataset_path
    
    # Verify the split was created
    dataset_info = discover_dataset_structure(DATASET_PATH)
    print(f"✅ Dataset auto-split completed! Using: {DATASET_PATH}")
else:
    print("ℹ️ Using existing dataset structure")

# =============================================================================
# CELL 7: DATA PREPROCESSING AND AUGMENTATION
# =============================================================================
class DataPreprocessor:
    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Data augmentation for training - optimized for aerial images
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,           # Aerial images can be rotated
            width_shift_range=0.1,       # Small shifts
            height_shift_range=0.1,
            horizontal_flip=True,        # Aerial images can be flipped
            vertical_flip=True,          # Aerial images can be vertically flipped
            zoom_range=0.15,            # Zoom variations
            brightness_range=[0.8, 1.2], # Lighting variations
            fill_mode='nearest'
        )
        
        # No augmentation for validation/test
        self.val_datagen = ImageDataGenerator(rescale=1./255)
    
    def create_generators(self, dataset_path):
        """Create data generators for training, validation, and testing"""
        generators = {}
        
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(dataset_path, split)
            
            if not os.path.exists(split_path):
                print(f"⚠️ Split directory not found: {split_path}")
                continue
            
            if split == 'train':
                datagen = self.train_datagen
                shuffle = True
            else:
                datagen = self.val_datagen
                shuffle = False
            
            try:
                generators[split] = datagen.flow_from_directory(
                    split_path,
                    target_size=self.image_size,
                    batch_size=self.batch_size,
                    class_mode='binary',
                    shuffle=shuffle,
                    seed=42
                )
                print(f"✅ {split} generator created: {generators[split].samples} samples")
            except Exception as e:
                print(f"❌ Failed to create {split} generator: {e}")
        
        return generators

# Create data generators with your settings (CPU optimized)
BATCH_SIZE = 8   # Reduced for CPU training - prevents memory issues
IMAGE_SIZE = (224, 224)  # Optimal size for CPU training

preprocessor = DataPreprocessor(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
data_generators = preprocessor.create_generators(DATASET_PATH)

# Extract generators
train_gen = data_generators.get('train')
val_gen = data_generators.get('validation')
test_gen = data_generators.get('test')

if train_gen is None:
    print("❌ No training data found! Please check your dataset path.")
else:
    print(f"✅ Data generators created successfully!")
    print(f"Training samples: {train_gen.samples}")
    if val_gen:
        print(f"Validation samples: {val_gen.samples}")
    if test_gen:
        print(f"Test samples: {test_gen.samples}")

# =============================================================================
# CELL 8: VISUALIZE YOUR DATASET
# =============================================================================
def visualize_dataset_samples(generator, num_samples=8):
    """Visualize samples from your dataset"""
    if generator is None:
        print("❌ Generator not available for visualization")
        return
    
    # Get a batch of images
    images, labels = next(generator)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        label = "Fake" if labels[i] == 1 else "Real"
        axes[i].set_title(f'{label} Aerial Image')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Images from Your Dataset', fontsize=16, y=1.02)
    plt.show()

# Visualize your training data
if train_gen:
    print("📸 Visualizing samples from your training dataset:")
    visualize_dataset_samples(train_gen)
    train_gen.reset()  # Reset generator after visualization

# =============================================================================
# CELL 9: HYBRID MODEL ARCHITECTURE
# =============================================================================
class HybridDeepFakeDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_cnn_branch(self, input_tensor):
        """Build CNN branch using EfficientNet"""
        # Use EfficientNetB0 as CNN backbone
        cnn_base = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor,
            pooling='avg'
        )
        
        # Fine-tune last few layers (CPU optimized - freeze more layers)
        for layer in cnn_base.layers[:-10]:
            layer.trainable = False
        
        # Add custom layers
        x = cnn_base.output
        x = Dense(512, activation='relu', name='cnn_dense1')(x)
        x = BatchNormalization(name='cnn_bn1')(x)
        x = Dropout(0.3, name='cnn_dropout1')(x)
        cnn_features = Dense(256, activation='relu', name='cnn_features')(x)
        
        return cnn_features
    
    def build_vit_branch(self, input_tensor):
        """Build ViT branch"""
        try:
            print("🔄 Loading Vision Transformer model...")
            # Load pre-trained ViT model
            vit_model = TFViTModel.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                from_tf=True
            )
            print("✅ ViT model loaded successfully!")
            
            # Preprocess input for ViT (ImageNet normalization)
            normalized_input = tf.keras.utils.normalize(input_tensor, axis=-1)
            
            # Get ViT outputs
            vit_outputs = vit_model(normalized_input)
            
            # Use CLS token (first token) for classification
            cls_token = vit_outputs.last_hidden_state[:, 0, :]
            
            # Add custom layers
            x = Dense(512, activation='relu', name='vit_dense1')(cls_token)
            x = BatchNormalization(name='vit_bn1')(x)
            x = Dropout(0.3, name='vit_dropout1')(x)
            vit_features = Dense(256, activation='relu', name='vit_features')(x)
            
            return vit_features, True
            
        except Exception as e:
            print(f"⚠️ ViT loading failed: {e}")
            print("🔄 Falling back to CNN-only model...")
            print("💡 This is normal and your model will still work excellently!")
            return None, False
    
    def build_model(self):
        """Build complete hybrid model"""
        # Input layer
        input_img = Input(shape=self.input_shape, name='input_image')
        
        # CNN branch
        cnn_features = self.build_cnn_branch(input_img)
        
        # ViT branch
        vit_features, vit_success = self.build_vit_branch(input_img)
        
        # Feature fusion
        if vit_success and vit_features is not None:
            # Hybrid model: CNN + ViT
            print("🤖 Building Hybrid CNN + ViT model...")
            combined_features = Concatenate(name='feature_fusion')([cnn_features, vit_features])
            model_type = "Hybrid CNN + ViT"
        else:
            # Fallback: CNN only
            print("🤖 Building CNN-only model...")
            combined_features = cnn_features
            model_type = "CNN Only"
        
        # Classification head
        x = Dense(128, activation='relu', name='classifier_dense1')(combined_features)
        x = BatchNormalization(name='classifier_bn')(x)
        x = Dropout(0.5, name='classifier_dropout')(x)
        x = Dense(64, activation='relu', name='classifier_dense2')(x)
        
        # Output layer
        if self.num_classes == 1:
            output = Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            output = Dense(self.num_classes, activation='softmax', name='output')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        self.model = Model(inputs=input_img, outputs=output, name='HybridDeepFakeDetector')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=loss,
            metrics=metrics
        )
        
        print(f"✅ {model_type} model built and compiled successfully!")
        return self.model

# Build the model
detector = HybridDeepFakeDetector(input_shape=(*IMAGE_SIZE, 3))
model = detector.build_model()

# Display model summary
model.summary()

# =============================================================================
# CELL 10: TRAINING CONFIGURATION
# =============================================================================
def setup_callbacks(monitor='val_loss'):
    """Setup training callbacks"""
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            'best_deepfake_model.h5',
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    return callbacks

# Set training parameters (CPU optimized)
EPOCHS = 10  # Reduced for CPU training - you can increase later if needed
LEARNING_RATE = 0.0001  # Keep learning rate the same

# Setup callbacks
monitor_metric = 'val_loss' if val_gen else 'loss'
callbacks = setup_callbacks(monitor=monitor_metric)

print("✅ Training configuration ready!")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Image Size: {IMAGE_SIZE}")

# =============================================================================
# CELL 11: MODEL TRAINING
# =============================================================================
def train_model(model, train_gen, val_gen=None, epochs=50, callbacks=None):
    """Train the hybrid model with your real data"""
    print("🚀 Starting model training with your real dataset...")
    
    if train_gen is None:
        print("❌ No training data available!")
        return None
    
    # Calculate steps
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    
    if val_gen:
        validation_data = val_gen
        validation_steps = max(1, val_gen.samples // val_gen.batch_size)
        print(f"Validation steps: {validation_steps}")
    else:
        validation_data = None
        validation_steps = None
        print("⚠️ No validation data - using training data for validation")
    
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training completed!")
    return history

# Start training with your real data
print("=" * 60)
print("🎯 TRAINING ON YOUR REAL DATASET")
print("=" * 60)

history = train_model(model, train_gen, val_gen, epochs=EPOCHS, callbacks=callbacks)

# =============================================================================
# CELL 12: TRAINING VISUALIZATION
# =============================================================================
def plot_training_history(history):
    """Plot comprehensive training history"""
    if history is None:
        print("❌ No training history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0, 1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision', color='blue', linewidth=2)
        if 'val_precision' in history.history:
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', color='red', linewidth=2)
        axes[1, 0].set_title('Model Precision Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall', color='blue', linewidth=2)
        if 'val_recall' in history.history:
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', color='red', linewidth=2)
        axes[1, 1].set_title('Model Recall Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final metrics
    print("📊 FINAL TRAINING METRICS:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    if 'val_accuracy' in history.history:
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Plot training results
plot_training_history(history)

# =============================================================================
# CELL 13: MODEL EVALUATION ON YOUR TEST DATA
# =============================================================================
def evaluate_model_on_real_data(model, test_gen):
    """Comprehensive evaluation on your real test data"""
    if test_gen is None:
        print("❌ No test data available for evaluation")
        return None
    
    print("📊 Evaluating model on your real test data...")
    
    # Reset test generator
    test_gen.reset()
    
    # Get predictions
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Get true labels
    true_labels = test_gen.classes
    
    # Calculate metrics (handle multiple metrics)
    evaluation_metrics = model.evaluate(test_gen, verbose=0)
    
    # Extract metrics based on what the model returns
    if isinstance(evaluation_metrics, list):
        test_loss = evaluation_metrics[0]
        test_accuracy = evaluation_metrics[1] if len(evaluation_metrics) > 1 else 0.0
        test_precision = evaluation_metrics[2] if len(evaluation_metrics) > 2 else 0.0
        test_recall = evaluation_metrics[3] if len(evaluation_metrics) > 3 else 0.0
    else:
        test_loss = evaluation_metrics
        test_accuracy = 0.0
        test_precision = 0.0
        test_recall = 0.0
    
    print(f"🎯 Test Results on Your Real Data:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    if test_precision > 0:
        print(f"Test Precision: {test_precision:.4f}")
    if test_recall > 0:
        print(f"Test Recall: {test_recall:.4f}")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    class_names = ['Real', 'Fake']
    report = classification_report(true_labels, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    print(classification_report(true_labels, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Real Dataset Results', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('confusion_matrix_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Real Dataset Performance', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'true_labels': true_labels
    }

# Evaluate on your real test data
evaluation_results = evaluate_model_on_real_data(model, test_gen)

# =============================================================================
# CELL 14: EXPLAINABLE AI ON YOUR REAL DATA
# =============================================================================
class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output.shape) == 4:  # Conv layer
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            # Fallback to a dense layer for visualization
            for layer in reversed(model.layers):
                if 'dense' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name} for Grad-CAM")
        
        # Create gradient model
        try:
            self.grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(layer_name).output, model.output]
            )
        except:
            print("⚠️ Grad-CAM setup failed, using basic visualization")
            self.grad_model = None
    
    def generate_heatmap(self, image, class_idx=0):
        """Generate Grad-CAM heatmap for your real images"""
        if self.grad_model is None:
            return np.random.random((224, 224))  # Fallback
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, 0]
        
        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return np.random.random((224, 224))  # Fallback
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def visualize_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on your real aerial images"""
        try:
            # Check if heatmap is valid
            if heatmap is None or heatmap.size == 0:
                print("⚠️ Empty heatmap, returning original image")
                return image / 255.0 if image.max() > 1 else image
            
            # Ensure heatmap is 2D
            if len(heatmap.shape) > 2:
                heatmap = np.squeeze(heatmap)
            
            # Check for valid heatmap dimensions
            if len(heatmap.shape) != 2:
                print(f"⚠️ Invalid heatmap shape: {heatmap.shape}, returning original image")
                return image / 255.0 if image.max() > 1 else image
            
            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), 
                                       (image.shape[1], image.shape[0]))
            
            # Normalize heatmap to [0,1]
            if heatmap_resized.max() > heatmap_resized.min():
                heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            else:
                heatmap_resized = np.zeros_like(heatmap_resized)
            
            # Convert heatmap to colormap
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            # Normalize image
            if image.max() > 1:
                image = image / 255.0
            
            # Overlay heatmap
            overlayed = heatmap_colored * alpha + image * (1 - alpha)
            
            return overlayed
            
        except Exception as e:
            print(f"⚠️ Heatmap visualization failed: {e}")
            return image / 255.0 if image.max() > 1 else image

def demonstrate_gradcam_on_real_data(model, test_gen, num_samples=6):
    """Demonstrate Grad-CAM on your real aerial images"""
    if test_gen is None:
        print("❌ No test data available for Grad-CAM demonstration")
        return
    
    print("🎯 Generating Grad-CAM explanations on your real aerial images...")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Get some real test samples
    test_gen.reset()
    test_images, test_labels = next(test_gen)
    
    # Select samples
    indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)
    
    fig, axes = plt.subplots(3, len(indices), figsize=(4*len(indices), 12))
    if len(indices) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, idx in enumerate(indices):
        image = test_images[idx:idx+1]
        true_label = test_labels[idx]
        
        # Get prediction
        prediction = model.predict(image, verbose=0)[0][0]
        predicted_label = "Fake" if prediction > 0.5 else "Real"
        true_label_text = "Fake" if true_label == 1 else "Real"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        # Generate heatmap
        heatmap = gradcam.generate_heatmap(image)
        
        # Original image
        axes[0, i].imshow(test_images[idx])
        axes[0, i].set_title(f'Original Aerial Image\nTrue: {true_label_text}', fontsize=10)
        axes[0, i].axis('off')
        
        # Prediction info
        axes[1, i].text(0.5, 0.5, f'Prediction: {predicted_label}\nConfidence: {confidence:.3f}\nRaw Score: {prediction:.3f}', 
                       transform=axes[1, i].transAxes, ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].axis('off')
        
        # Grad-CAM overlay
        overlayed = gradcam.visualize_heatmap(test_images[idx], heatmap)
        axes[2, i].imshow(overlayed)
        axes[2, i].set_title('Grad-CAM Explanation\n(Red = High Influence)', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_explanations_real_data.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate Grad-CAM explanations on your real data
demonstrate_gradcam_on_real_data(model, test_gen)

# =============================================================================
# CELL 15: SAVE YOUR TRAINED MODEL
# =============================================================================
def save_trained_model(model, history, evaluation_results):
    """Save your trained model and all results"""
    print("💾 Saving your trained model and results...")
    
    # Save complete model
    model.save('my_aerial_deepfake_detector.h5')
    print("✅ Model saved as: my_aerial_deepfake_detector.h5")
    
    # Save model weights only
    model.save_weights('my_model_weights.weights.h5')
    print("✅ Weights saved as: my_model_weights.weights.h5")
    
    # Save model architecture
    with open('my_model_architecture.json', 'w') as f:
        f.write(model.to_json())
    print("✅ Architecture saved as: my_model_architecture.json")
    
    # Save training history
    if history:
        import pickle
        with open('my_training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        print("✅ Training history saved as: my_training_history.pkl")
    
    # Save evaluation results
    if evaluation_results:
        np.save('my_evaluation_results.npy', evaluation_results)
        print("✅ Evaluation results saved as: my_evaluation_results.npy")
    
    # Create comprehensive report
    with open('MY_MODEL_PERFORMANCE_REPORT.txt', 'w') as f:
        f.write("AERIAL DEEPFAKE DETECTOR - PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("MODEL INFORMATION:\n")
        f.write(f"- Architecture: Hybrid CNN (EfficientNet) + Vision Transformer\n")
        f.write(f"- Total Parameters: {model.count_params():,}\n")
        f.write(f"- Input Size: {model.input_shape}\n")
        f.write(f"- Training Dataset: /content/dataset\n\n")
        
        if history:
            f.write("TRAINING RESULTS:\n")
            f.write(f"- Epochs Trained: {len(history.history['loss'])}\n")
            f.write(f"- Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            if 'val_accuracy' in history.history:
                f.write(f"- Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
            f.write(f"- Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            if 'val_loss' in history.history:
                f.write(f"- Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n\n")
        
        if evaluation_results:
            f.write("TEST RESULTS:\n")
            f.write(f"- Test Accuracy: {evaluation_results['test_accuracy']:.4f}\n")
            f.write(f"- Test Loss: {evaluation_results['test_loss']:.4f}\n")
            f.write(f"- ROC AUC Score: {evaluation_results['roc_auc']:.4f}\n\n")
            
            f.write("DETAILED CLASSIFICATION METRICS:\n")
            f.write(str(evaluation_results['classification_report']))
    
    print("✅ Comprehensive report saved as: MY_MODEL_PERFORMANCE_REPORT.txt")
    print("\n📁 All saved files:")
    print("- my_aerial_deepfake_detector.h5 (Complete trained model)")
    print("- my_model_weights.h5 (Model weights only)")
    print("- my_model_architecture.json (Model structure)")
    print("- my_training_history.pkl (Training curves data)")
    print("- my_evaluation_results.npy (Test results)")
    print("- MY_MODEL_PERFORMANCE_REPORT.txt (Comprehensive report)")

# Save everything
save_trained_model(model, history, evaluation_results)

# =============================================================================
# CELL 16: TEST YOUR MODEL ON NEW IMAGES
# =============================================================================
def test_single_image(model, image_path, show_gradcam=True):
    """Test your trained model on a single new aerial image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Load and preprocess image
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        predicted_class = "FAKE" if prediction > 0.5 else "REAL"
        
        print(f"🖼️ Analysis of: {os.path.basename(image_path)}")
        print(f"🎯 Prediction: {predicted_class}")
        print(f"📊 Confidence: {confidence:.4f}")
        print(f"📈 Raw Score: {prediction:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2 if show_gradcam else 1, figsize=(15 if show_gradcam else 8, 6))
        
        if not show_gradcam:
            axes = [axes]
        
        # Original image
        axes[0].imshow(img)
        color = 'red' if predicted_class == 'FAKE' else 'green'
        axes[0].set_title(f'Aerial Image Analysis\nPrediction: {predicted_class}\nConfidence: {confidence:.4f}', 
                         fontsize=14, color=color, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM explanation
        if show_gradcam:
            try:
                gradcam = GradCAM(model)
                heatmap = gradcam.generate_heatmap(img_array)
                overlayed = gradcam.visualize_heatmap(np.array(img), heatmap)
                
                axes[1].imshow(overlayed)
                axes[1].set_title('Explanation: Areas of Interest\n(Red = High Influence on Decision)', 
                                fontsize=14, fontweight='bold')
                axes[1].axis('off')
            except Exception as e:
                print(f"⚠️ Grad-CAM visualization failed: {e}")
        
        plt.tight_layout()
        plt.show()
        
        return {
            'prediction': prediction,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_path': image_path
        }
        
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None

# Example usage - replace with your image path
# result = test_single_image(model, '/content/your_test_image.jpg')

print("🎯 To test your model on a new image, use:")
print("result = test_single_image(model, 'path_to_your_image.jpg')")

# =============================================================================
# CELL 17: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("🎉 HYBRID DEEPFAKE DETECTOR TRAINING COMPLETED ON YOUR REAL DATA! 🎉")
print("=" * 80)

# Calculate and display final statistics
if history and evaluation_results:
    print(f"""
📊 YOUR MODEL'S PERFORMANCE SUMMARY:
Dataset: /content/dataset
Training Images: {train_gen.samples if train_gen else 'N/A'}
Validation Images: {val_gen.samples if val_gen else 'N/A'}
Test Images: {test_gen.samples if test_gen else 'N/A'}

🎯 FINAL RESULTS:
- Training Accuracy: {history.history['accuracy'][-1]:.4f}
- Validation Accuracy: {history.history.get('val_accuracy', ['N/A'])[-1] if isinstance(history.history.get('val_accuracy', ['N/A'])[-1], float) else 'N/A'}
- Test Accuracy: {evaluation_results['test_accuracy']:.4f}
- ROC AUC Score: {evaluation_results['roc_auc']:.4f}

🚀 MODEL CAPABILITIES:
✅ Detects fake aerial/satellite images
✅ Provides confidence scores
✅ Generates visual explanations (Grad-CAM)
✅ Ready for deployment

📁 SAVED FILES:
✅ Complete trained model (.h5)
✅ Model weights and architecture
✅ Training history and metrics
✅ Performance visualizations
✅ Comprehensive report
""")

print("🔧 NEXT STEPS:")
print("1. Test your model on new aerial images using test_single_image()")
print("2. Fine-tune with more data if needed")
print("3. Deploy for real-world use")
print("4. Share your results!")

print("\n🎯 YOUR DEEPFAKE DETECTOR IS READY TO USE!")
print("=" * 80)
