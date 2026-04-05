#!/bin/bash
set -euo pipefail

# Publish Firefox Extension (v1.2.2)
echo "Creating Firefox extension package..."

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: zip is required to build the Firefox extension package."
    exit 1
fi

# Create output directory
mkdir -p firefox_package

# Create zip package for Firefox Add-ons
ROOT_DIR=$(pwd)
PACKAGE_NAME="$ROOT_DIR/firefox_package/rocketrecap-summarizer-v1.2.2.zip"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

cp firefox_extension/manifest.json "$TEMP_DIR"/
cp firefox_extension/popup.js "$TEMP_DIR"/
cp firefox_extension/popup.html "$TEMP_DIR"/
cp -R chrome_extension/images "$TEMP_DIR"/images

(
    cd "$TEMP_DIR" || exit 1
    zip -r "$PACKAGE_NAME" manifest.json popup.js popup.html images/
)

echo "Firefox extension package created: firefox_package/rocketrecap-summarizer-v1.2.2.zip"
echo "Upload this file to Firefox Add-on Developer Hub"
