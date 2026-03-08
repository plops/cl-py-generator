#!/bin/bash

# Publish Firefox Extension (v1.2.2)
echo "Creating Firefox extension package..."

# Create output directory
mkdir -p firefox_package

cd firefox_extension

# Ensure images directory exists
if [ ! -d "images" ]; then
    echo "Error: images directory not found in firefox_extension/"
    exit 1
fi

# Create zip package for Firefox Add-ons
PACKAGE_NAME="../firefox_package/rocketrecap-summarizer-v1.2.2.zip"
zip -r "$PACKAGE_NAME" manifest.json popup.js popup.html images/

echo "Firefox extension package created: firefox_package/rocketrecap-summarizer-v1.2.2.zip"
echo "Upload this file to Firefox Add-on Developer Hub"
