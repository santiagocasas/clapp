#! /bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Pipeline fails on any command failure

echo "=== Starting CLASS installation process ==="
echo "Current directory: $(pwd)"
echo "Contents of directory: $(ls -la)"

# Git clone class only if the directory doesn't exist yet
if [ ! -d "class_public" ]; then
    echo "=== Cloning CLASS repository ==="
    git clone --depth 1 --single-branch --branch master https://github.com/lesgourg/class_public.git
else
    echo "=== CLASS repository already exists, skipping clone ==="
fi

# Change to the class_public directory
echo "=== Changing to class_public directory ==="
if [ ! -d "class_public" ]; then
    echo "ERROR: class_public directory not found!"
    exit 1
fi

cd class_public/
echo "Now in: $(pwd)"
echo "Contents of class_public: $(ls -la)"

# Clean previous builds
echo "=== Cleaning previous build ==="
make clean || echo "Clean step failed, but continuing..."

# Install dependencies
echo "=== Installing Python dependencies ==="
echo "Installing numpy..."
pip install "numpy<=1.26" -v || { echo "Failed to install numpy"; exit 1; }

echo "Installing scipy..."
pip install "scipy" -v || { echo "Failed to install scipy"; exit 1; }

echo "Installing Cython..."
pip install "Cython" -v || { echo "Failed to install Cython"; exit 1; }

# Build CLASS
echo "=== Building CLASS library ==="
make -j || { echo "Failed to build CLASS"; exit 1; }

# Install the Python module
#echo "=== Installing CLASS Python module ==="
#pip install . -v || { echo "Failed to install CLASS Python module"; exit 1; }

# Verify installation
#echo "=== Verifying installation ==="
#python3 -c "from classy import Class; print('CLASS successfully imported!')" || { echo "Failed to import CLASS in Python"; exit 1; }

echo "=== CLASS installation completed successfully ==="
exit 0
