#!/usr/bin/env python3
"""
Simple script to run the Aerial DeepFake Detection Web Application
"""

import os
import sys

def check_model_file():
    """Check if the trained model file exists"""
    model_path = 'my_aerial_deepfake_detector.h5'
    if not os.path.exists(model_path):
        print("❌ ERROR: Model file not found!")
        print(f"Looking for: {model_path}")
        print("\n📥 To get your model from Google Colab:")
        print("1. In your Colab notebook, run:")
        print("   from google.colab import files")
        print("   files.download('my_aerial_deepfake_detector.h5')")
        print("2. Copy the downloaded file to this directory")
        print("3. Run this script again")
        return False
    return True

def check_directories():
    """Create required directories if they don't exist"""
    directories = ['uploads', 'static/results', 'static/temp']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")

def main():
    print("🚀 Aerial DeepFake Detection Web Application")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    print("✅ Model file found")
    
    # Create directories
    check_directories()
    
    # Check if requirements are installed
    try:
        import flask
        import tensorflow
        import cv2
        import numpy
        import PIL
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Import and run the app
    try:
        from app import app
        print("\n🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
