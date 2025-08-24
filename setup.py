"""
Setup script for Aerial DeepFake Detection Web Application
This script helps set up the environment and dependencies
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Print setup header"""
    print("🚀 Aerial DeepFake Detection - Setup Script")
    print("=" * 60)
    print("This script will help you set up the web application")
    print()

def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip not found")
        return False

def create_virtual_env():
    """Create virtual environment"""
    print("🔧 Setting up virtual environment...")
    
    venv_name = "venv"
    if os.path.exists(venv_name):
        print(f"⚠️ Virtual environment '{venv_name}' already exists")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_name)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"✅ Virtual environment '{venv_name}' created")
        
        # Provide activation instructions
        if platform.system() == "Windows":
            activation_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activation_cmd = f"source {venv_name}/bin/activate"
        
        print(f"💡 To activate: {activation_cmd}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_requirements():
    """Install Python requirements"""
    print("📥 Installing Python packages...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if not in_venv:
            print("⚠️ Not in virtual environment")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return False
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("📁 Creating directories...")
    
    directories = [
        "uploads",
        "static",
        "static/css",
        "static/results", 
        "static/temp",
        "templates"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"ℹ️ Already exists: {directory}")

def check_model_file():
    """Check for trained model file"""
    print("🤖 Checking for trained model...")
    
    model_path = "my_aerial_deepfake_detector.h5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("\n📥 To get your model from Google Colab:")
        print("1. In your Colab notebook, run:")
        print("   from google.colab import files")
        print("   files.download('my_aerial_deepfake_detector.h5')")
        print("2. Copy the downloaded file to this directory")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    packages = [
        ("flask", "Flask"),
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib")
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Ensure your trained model file is in this directory:")
    print("   my_aerial_deepfake_detector.h5")
    print()
    print("2. Run the application:")
    print("   python run.py")
    print("   or")
    print("   python app.py")
    print()
    print("3. Open your browser and go to:")
    print("   http://localhost:5000")
    print()
    print("🎯 Happy deepfake detection!")

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup environment
    create_directories()
    
    # Ask about virtual environment
    response = input("Create virtual environment? (recommended) (y/n): ").lower()
    if response == 'y':
        if not create_virtual_env():
            print("⚠️ Continuing without virtual environment")
    
    # Install requirements
    response = input("Install Python packages? (y/n): ").lower()
    if response == 'y':
        if not install_requirements():
            print("❌ Package installation failed")
            sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("⚠️ Some packages are missing - install requirements.txt")
    
    # Check model
    check_model_file()
    
    # Finish
    print_next_steps()

if __name__ == "__main__":
    main()
