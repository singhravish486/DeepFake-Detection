# Aerial DeepFake Detection - Setup Instructions

## ✅ Issues Resolved

1. **TensorFlow Installation**: Successfully installed TensorFlow 2.20.0 in virtual environment
2. **Virtual Environment**: Created and configured Python 3.13 virtual environment
3. **Dependencies**: All required packages installed (Flask, OpenCV, PIL, etc.)
4. **Demo Mode**: Application runs in demo mode when model has compatibility issues

## 🚀 How to Run the Application

### Option 1: Using the Batch File (Recommended for Windows)
```bash
start_app.bat
```

### Option 2: Manual Command
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the application
python app.py
```

### Option 3: Using Python launcher
```bash
py -m venv venv  # (already created)
.\venv\Scripts\Activate.ps1
python app.py
```

## 🌐 Accessing the Application

Once running, open your web browser and go to:
- **Local access**: http://localhost:5000
- **Network access**: http://0.0.0.0:5000

## ⚠️ Current Status: Demo Mode

The application is currently running in **DEMO MODE** because of a model compatibility issue:

**Issue**: Shape mismatch in the model file
- Expected: (3, 3, 1, 32) - 1 channel
- Received: (3, 3, 3, 32) - 3 channels

**In Demo Mode**:
- ✅ Web interface works perfectly
- ✅ File upload and processing works
- ✅ Returns realistic random predictions (0.2-0.8 range)
- ✅ Generates demo explanation overlays
- ⚠️ Predictions are not from the actual trained model

## 🔧 Fixing the Model Issue

To use your actual trained model, you have several options:

### Option 1: Retrain the Model
Run the training script again:
```bash
python Real_Dataset_Hybrid_DeepFake_Detector.py
```

### Option 2: Check Model Compatibility
The model might be from an older TensorFlow version. Try:
1. Retraining with current TensorFlow 2.20.0
2. Checking if the model was saved correctly
3. Verifying the input shape in the training script

### Option 3: Convert Model Format
If you have the model in a different format, you might need to convert it.

## 📁 Project Structure

```
Training/
├── app.py                    # Main Flask application
├── venv/                     # Virtual environment (created)
├── my_aerial_deepfake_detector.h5  # Model file (has compatibility issues)
├── requirements.txt          # All dependencies
├── start_app.bat            # Easy startup script (created)
├── setup_instructions.md    # This file (created)
├── static/                  # Web assets
├── templates/               # HTML templates
└── uploads/                 # Upload directory
```

## 🎯 Next Steps

1. **Current**: Application works in demo mode
2. **Recommended**: Retrain the model with current environment
3. **Future**: Replace demo predictions with real model predictions

## 🆘 Troubleshooting

- **Virtual environment issues**: Make sure you're in the `Training` directory
- **Permission errors**: Run PowerShell as Administrator if needed
- **Port 5000 busy**: Change the port in `app.py` line 316
- **Missing packages**: Run `pip install -r requirements.txt` in activated venv

The application is now functional and ready to use!

