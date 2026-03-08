#!/bin/bash

# Publish Chrome Extension (v1.2.3)
echo "Creating Chrome extension package..."

# Create output directory
mkdir -p chrome_package

cd chrome_extension

# Ensure images directory exists
if [ ! -d "images" ]; then
    echo "Error: images directory not found in chrome_extension/"
    exit 1
fi

# Create zip package for Chrome Web Store
PACKAGE_NAME="../chrome_package/rocketrecap-summarizer-v1.2.3.zip"
zip -r "$PACKAGE_NAME" manifest.json popup.js popup.html images/

echo "Chrome extension package created: chrome_package/rocketrecap-summarizer-v1.2.3.zip"
echo "Upload this file to Chrome Web Store Developer Dashboard"
