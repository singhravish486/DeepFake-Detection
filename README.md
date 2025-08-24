# Aerial DeepFake Detection Web Application

A professional web application for detecting manipulated aerial and satellite images using advanced AI with explainable decision-making capabilities.

## 🎯 Features

- **Hybrid AI Model**: Combines CNN (EfficientNet) and Vision Transformer architectures
- **Explainable AI**: Grad-CAM visualizations show which parts influenced the decision
- **Professional UI**: Modern, responsive web interface with Bootstrap
- **Real-time Analysis**: Upload and analyze images instantly
- **Comprehensive Results**: Detailed predictions with confidence scores
- **Mobile Friendly**: Responsive design works on all devices

## 🏗️ Architecture

### Deep Learning Model
- **CNN Branch**: EfficientNetB0 for local feature analysis
- **ViT Branch**: Vision Transformer for global context (with fallback)
- **Feature Fusion**: Intelligent combination of both approaches
- **Transfer Learning**: Pre-trained weights for better performance

### Web Application
- **Backend**: Flask with TensorFlow integration
- **Frontend**: Bootstrap 5 with modern CSS and JavaScript
- **File Handling**: Secure upload with validation
- **Results Display**: Interactive visualizations and explanations

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 4GB RAM (8GB recommended)
- Modern web browser

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir aerial-deepfake-detection
cd aerial-deepfake-detection

# Copy all the web application files here
# (app.py, templates/, static/, requirements.txt, etc.)
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Add Your Trained Model

Copy your trained model from Google Colab to the project directory:

```bash
# The model file should be named exactly:
my_aerial_deepfake_detector.h5
```

**To get your model from Google Colab:**
1. In your Colab notebook, run:
   ```python
   from google.colab import files
   files.download('my_aerial_deepfake_detector.h5')
   ```
2. Copy the downloaded file to your project directory

### 4. Create Required Directories

```bash
mkdir uploads
mkdir static/results
mkdir static/temp
```

### 5. Run the Application

```bash
python app.py
```

The application will start at: `http://localhost:5000`

## 📁 Project Structure

```
aerial-deepfake-detection/
├── app.py                          # Main Flask application
├── my_aerial_deepfake_detector.h5   # Your trained model (add this)
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── templates/
│   ├── base.html                   # Base template
│   ├── index.html                  # Home page
│   ├── analyze.html                # Analysis page
│   └── about.html                  # About page
├── static/
│   ├── css/
│   │   └── style.css              # Custom CSS styles
│   ├── results/                   # Generated result images
│   └── temp/                      # Temporary files
└── uploads/                       # Uploaded images
```

## 🎮 How to Use

### 1. Home Page
- Overview of the system and features
- Navigation to analysis page

### 2. Analyze Page
- **Upload Image**: Drag & drop or click to browse
- **Supported Formats**: JPG, PNG, GIF, BMP, TIFF
- **Max File Size**: 16MB
- **Get Results**: Instant analysis with explanations

### 3. Results Display
- **Prediction**: Real or Fake classification
- **Confidence Score**: How certain the AI is
- **Probabilities**: Detailed breakdown
- **Grad-CAM**: Visual explanation showing influential regions
- **Interactive Tabs**: Switch between original and explained images

### 4. About Page
- Technical details about the system
- Architecture explanation
- Applications and use cases

## 🔧 Configuration

### Model Settings
Edit `app.py` to modify:

```python
MODEL_PATH = 'my_aerial_deepfake_detector.h5'  # Model file path
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size
```

### Upload Settings
```python
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
```

### Allowed File Types
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment
The application is ready for deployment on:
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **Google Cloud Platform**: Use App Engine
- **AWS**: Deploy on Elastic Beanstalk
- **Azure**: Use App Service

## 🛠️ Troubleshooting

### Common Issues

**1. Model Not Found**
```
❌ Model file not found: my_aerial_deepfake_detector.h5
```
**Solution**: Copy your trained model file to the project directory

**2. Memory Issues**
```
OOM when allocating tensor
```
**Solution**: Reduce batch size or use smaller images

**3. Import Errors**
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Install requirements with `pip install -r requirements.txt`

**4. Upload Issues**
```
413 Request Entity Too Large
```
**Solution**: Check file size limit in `app.py`

### Performance Optimization

**For Better Performance:**
- Use GPU-enabled TensorFlow if available
- Implement image caching
- Add Redis for session management
- Use CDN for static files

**For Production:**
- Enable HTTPS
- Add rate limiting
- Implement user authentication
- Add logging and monitoring

## 📊 Model Performance

### Current Status
- **Architecture**: Hybrid CNN + ViT
- **Training Data**: Limited dataset (demonstration)
- **Current Accuracy**: ~50% (needs more training data)
- **Expected with Full Data**: 85-95% accuracy

### Improving Performance
1. **Collect More Data**: 1000+ images per class
2. **Diverse Sources**: Multiple types of aerial imagery
3. **Better Augmentation**: Advanced data augmentation techniques
4. **Ensemble Methods**: Combine multiple models

## 🔬 Technical Details

### Model Architecture
- **Base Model**: EfficientNetB0
- **Input Size**: 224×224×3
- **Parameters**: ~4.9M total (1.7M trainable)
- **Framework**: TensorFlow 2.x
- **Training**: Transfer learning with fine-tuning

### Explainable AI
- **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Purpose**: Show which image regions influenced the decision
- **Visualization**: Heatmap overlay on original image
- **Colors**: Red = high influence, Blue = low influence

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your local laws and regulations when using this technology.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For questions or issues, please create an issue in the repository or contact the development team.

---

**Built with ❤️ using TensorFlow, Flask, and Bootstrap**
