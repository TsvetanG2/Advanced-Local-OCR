#!/usr/bin/env python3
"""
Setup script for OCR Document Processor.
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ocr-document-processor",
    version="2.0.0",
    author="OCR Solutions",
    author_email="contact@ocrsolutions.com",
    description="Advanced OCR processing application with LLM correction capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ocrsolutions/ocr-document-processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "llm": ["openai>=1.0.0", "anthropic>=0.7.0"],
        "export": ["pandas>=1.3.0", "openpyxl>=3.0.9", "reportlab>=3.6.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
    },
    entry_points={
        "console_scripts": [
            "ocr-processor=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    keywords="ocr, text-recognition, image-processing, llm, ai, document-processing",
    project_urls={
        "Bug Reports": "https://github.com/ocrsolutions/ocr-document-processor/issues",
        "Source": "https://github.com/ocrsolutions/ocr-document-processor",
        "Documentation": "https://github.com/ocrsolutions/ocr-document-processor/wiki",
    },
)
