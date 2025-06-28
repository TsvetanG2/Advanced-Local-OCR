# Installation Guide

## 🚀 **Quick Installation**

### **Method 1: Standard Installation (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/advanced-local-ocr-studio.git
cd advanced-local-ocr-studio

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install OCR engines
pip install easyocr pytesseract

# 4. Install Tesseract (system dependency)
# See platform-specific instructions below

# 5. Run the application
python app.py
```

### **Method 2: Virtual Environment (Recommended for Development)**

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/advanced-local-ocr-studio.git
cd advanced-local-ocr-studio

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install easyocr pytesseract

# 5. Run application
python app.py
```

## 🔧 **Platform-Specific Setup**

### **Windows**

#### **Tesseract Installation:**
1. Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (choose "Add to PATH" option)
3. Verify installation: `tesseract --version`

#### **Python Setup:**
```cmd
# Install Python 3.8+ from python.org
# Then install dependencies
pip install -r requirements.txt
pip install easyocr pytesseract
```

### **macOS**

#### **Using Homebrew (Recommended):**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract
brew install tesseract

# Install Python dependencies
pip install -r requirements.txt
pip install easyocr pytesseract
```

#### **Using MacPorts:**
```bash
sudo port install tesseract
pip install -r requirements.txt
pip install easyocr pytesseract
```

### **Linux (Ubuntu/Debian)**

```bash
# Update package list
sudo apt update

# Install Tesseract
sudo apt install tesseract-ocr

# Install Python dependencies
pip install -r requirements.txt
pip install easyocr pytesseract

# Optional: Install additional language packs
sudo apt install tesseract-ocr-eng  # English (usually included)
```

### **Linux (CentOS/RHEL/Fedora)**

```bash
# CentOS/RHEL:
sudo yum install tesseract

# Fedora:
sudo dnf install tesseract

# Install Python dependencies
pip install -r requirements.txt
pip install easyocr pytesseract
```

## 🧪 **Verify Installation**

### **Test Script**
```bash
# Run the installation test
python test_installation.py
```

### **Manual Verification**
```python
# Test in Python console
import cv2
import easyocr
import pytesseract
from PyQt5.QtWidgets import QApplication

print("✅ All dependencies imported successfully!")
```

### **Test OCR Engines**
```bash
# Test Tesseract
tesseract --version

# Test with sample image
python test_specific_text.py path/to/test/image.jpg
```

## ⚠️ **Troubleshooting**

### **Common Issues**

#### **"Tesseract not found"**
```bash
# Windows: Add Tesseract to PATH
# Default location: C:\Program Files\Tesseract-OCR
# Add to PATH environment variable

# macOS/Linux: Reinstall Tesseract
brew reinstall tesseract  # macOS
sudo apt reinstall tesseract-ocr  # Ubuntu
```

#### **"EasyOCR CUDA errors"**
```bash
# Install CPU-only version
pip uninstall easyocr
pip install easyocr --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### **"PyQt5 installation fails"**
```bash
# Try alternative installation
pip install PyQt5 --no-cache-dir

# Or use conda
conda install pyqt

# Ubuntu: Install system packages
sudo apt install python3-pyqt5
```

#### **"Permission denied" errors**
```bash
# Use user installation
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### **Performance Issues**

#### **Slow OCR Processing**
- Ensure images are not too large (resize if needed)
- Try disabling GPU for EasyOCR if causing issues
- Use preprocessing to improve image quality

#### **Memory Issues**
- Close other applications
- Process images one at a time
- Reduce image resolution

## 🎯 **Optional Enhancements**

### **GPU Acceleration (NVIDIA)**
```bash
# Install CUDA-enabled PyTorch for EasyOCR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Additional Export Formats**
```bash
# For Excel export
pip install pandas openpyxl

# For PDF export
pip install reportlab

# For advanced image processing
pip install scikit-image
```

### **Development Tools**
```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

## 📋 **System Requirements**

### **Minimum Requirements**
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04 (or equivalent)
- **Python**: 3.8 or higher
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 2 GB free space
- **Display**: 1024x768 resolution

### **Recommended Requirements**
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or higher
- **RAM**: 8 GB or more
- **Storage**: 5 GB free space
- **Display**: 1920x1080 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional)

## 🆘 **Getting Help**

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Run the test script**: `python test_installation.py`
3. **Search existing issues** on GitHub
4. **Create a new issue** with detailed information:
   - Operating system and version
   - Python version
   - Error messages
   - Steps to reproduce

## ✅ **Next Steps**

After successful installation:

1. **Run the application**: `python app.py`
2. **Load a test image** to verify OCR functionality
3. **Check the README** for usage instructions
4. **Explore the features** and settings

Enjoy using Advanced Local OCR Studio! 🎉
