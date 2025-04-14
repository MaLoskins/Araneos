#!/bin/bash

# PyTorch Reset and Blackwell (sm_120) Patch Installer
# For Python 3.10.11 and RTX 5080 GPUs

set -e  # Exit on error

echo "=========================================================="
echo "🧹 PYTORCH ENVIRONMENT RESET & BLACKWELL PATCH INSTALLER 🚀"
echo "=========================================================="
echo ""
echo "This script will:"
echo "  1. Reset your Python 3.10.11 packages (preserving Python itself)"
echo "  2. Install PyTorch from source with Blackwell (sm_120) support"
echo "  3. Verify the installation"
echo ""
echo "WARNING: This will remove all existing pip packages in your global environment!"
echo "Press CTRL+C now to abort, or Enter to continue..."
read

# Determine pip executable for Python 3.10.11
PYTHON_EXE="python"
PIP_EXE="pip"

# Check if Python 3.10.11 is available
PY_VERSION=$($PYTHON_EXE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
if [ "$PY_VERSION" != "3.10.11" ]; then
    echo "❌ ERROR: Found Python $PY_VERSION, but expected 3.10.11."
    echo "Please ensure Python 3.10.11 is in your PATH and is the default."
    exit 1
fi

echo "✅ Found Python 3.10.11"

# Check for CUDA 12.8
echo "Checking CUDA version..."
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  WARNING: NVCC not found. CUDA might not be installed correctly."
    echo "This script requires CUDA 12.8 to be installed."
    echo "Press Enter to continue anyway, or CTRL+C to abort..."
    read
else
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -ne 12 ] || [ "$CUDA_MINOR" -ne 8 ]; then
        echo "⚠️  WARNING: Found CUDA $CUDA_VERSION, but recommended version is 12.8."
        echo "This may cause compatibility issues with the Blackwell architecture."
        echo "Press Enter to continue anyway, or CTRL+C to abort..."
        read
    else
        echo "✅ Found CUDA $CUDA_VERSION"
    fi
fi

# Check for GPU
echo "Checking for RTX 5080 GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Cannot verify GPU model."
    echo "This script is designed for RTX 5080 GPUs."
    echo "Press Enter to continue anyway, or CTRL+C to abort..."
    read
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if [[ ! "$GPU_NAME" =~ "RTX 5080" ]]; then
        echo "⚠️  WARNING: RTX 5080 not detected. Found: $GPU_NAME"
        echo "This script is specifically for RTX 5080 GPUs with Blackwell architecture."
        echo "Press Enter to continue anyway, or CTRL+C to abort..."
        read
    else
        echo "✅ Found $GPU_NAME"
    fi
fi

# Create a work directory
WORKDIR=$(mktemp -d)
echo "Creating temporary work directory: $WORKDIR"
cd $WORKDIR

# Step 1: Reset Python packages while preserving Python itself
echo ""
echo "=========================================================="
echo "STEP 1: Resetting Python environment"
echo "=========================================================="

# List all packages before removal (for reference)
echo "Packages to be removed:"
$PIP_EXE list

# Save a list of installed packages (optional, for reference)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
$PIP_EXE freeze > "python_packages_backup_$TIMESTAMP.txt"
echo "📝 Package list saved to: $PWD/python_packages_backup_$TIMESTAMP.txt"

# More robust package uninstallation - handle one by one with filtering
echo "Uninstalling packages (this may take a while)..."

# First, try to clean up corrupted package directories
SITE_PACKAGES_DIR=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
echo "Looking for corrupted package directories in: $SITE_PACKAGES_DIR"

# This will find directories starting with a dash and attempt to remove them
find "$SITE_PACKAGES_DIR" -maxdepth 1 -type d -name "-*" | while read -r dir; do
    echo "Removing corrupted directory: $dir"
    rm -rf "$dir"
done

# Now handle package uninstallation more carefully
$PIP_EXE freeze | grep -v "^-e" | grep -v "@" | sed 's/==.*//' | grep -v "^-" | while read -r package; do
    if [[ -n "$package" && "$package" != "" ]]; then
        echo "Uninstalling: $package"
        $PIP_EXE uninstall -y "$package" || echo "Failed to uninstall $package - continuing anyway"
    fi
done

# Alternative approach to ensure all packages are removed
echo "Running pip uninstall with --all flag to remove any remaining packages"
$PIP_EXE uninstall --all -y || echo "pip uninstall --all may not be supported, continuing anyway"

# Verify packages are gone
echo "Remaining packages:"
$PIP_EXE list

# Install build dependencies
echo "Installing build dependencies..."
$PIP_EXE install -U pip setuptools wheel
$PIP_EXE install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Step 2: Clone repositories and apply patch
echo ""
echo "=========================================================="
echo "STEP 2: Cloning repositories and applying patch"
echo "=========================================================="

# Clone PyTorch repository
echo "Cloning PyTorch repository (this may take a while)..."
git clone --recursive https://github.com/pytorch/pytorch
if [ ! -d "pytorch" ]; then
    echo "❌ ERROR: Failed to clone PyTorch repository."
    exit 1
fi

# Clone patch repository
echo "Cloning patch repository..."
git clone https://github.com/kentstone84/pytorch-rtx5080-support.git
if [ ! -d "pytorch-rtx5080-support" ]; then
    echo "❌ ERROR: Failed to clone patch repository."
    exit 1
fi

# Apply the patch
echo "Applying Blackwell patch..."
cd pytorch
../pytorch-rtx5080-support/patch_blackwell.sh

# Step 3: Build and install PyTorch
echo ""
echo "=========================================================="
echo "STEP 3: Building and installing PyTorch"
echo "=========================================================="
echo "⚠️  This will take a significant amount of time (1-3 hours)."
echo "Press Enter to continue, or CTRL+C to abort..."
read

# Set environment variables for build
export TORCH_CUDA_ARCH_LIST="Blackwell"
export USE_CUDA=1
export USE_CUDNN=1
export BUILD_TEST=0
export MAX_JOBS=8  # Adjust based on your CPU cores

# Build and install PyTorch
echo "Building PyTorch with Blackwell support..."
$PYTHON_EXE setup.py install

# Step 4: Verify installation
echo ""
echo "=========================================================="
echo "STEP 4: Verifying installation"
echo "=========================================================="

echo "Creating verification script..."
cat > verify_pytorch.py << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA not available in PyTorch!")
    sys.exit(1)
    
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {properties.name}")
    print(f"  Compute capability: {properties.major}.{properties.minor}")
    print(f"  Total memory: {properties.total_memory / 1024**3:.2f} GB")
    
    # Check if Blackwell architecture is detected
    if properties.major == 12 and properties.minor == 0:
        print(f"✅ Blackwell architecture (sm_120) detected!")
    else:
        print(f"⚠️  Not using Blackwell architecture. Found: sm_{properties.major}{properties.minor}")

# Simple tensor operation test
try:
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print("\n✅ Basic CUDA tensor operation successful!")
except Exception as e:
    print(f"\n❌ CUDA tensor operation failed: {e}")
EOF

echo "Running verification..."
$PYTHON_EXE verify_pytorch.py

# Cleanup
echo ""
echo "=========================================================="
echo "STEP 5: Cleanup"
echo "=========================================================="

echo "Would you like to remove the build directories to save space? (y/n)"
read CLEANUP

if [[ "$CLEANUP" == "y" || "$CLEANUP" == "Y" ]]; then
    echo "Removing build directories..."
    cd ..
    rm -rf pytorch pytorch-rtx5080-support
    echo "Cleanup complete."
else
    echo "Build directories kept at: $WORKDIR"
fi

echo ""
echo "=========================================================="
echo "🎉 INSTALLATION COMPLETE!"
echo "=========================================================="
echo ""
echo "Your Python 3.10.11 environment has been reset and"
echo "PyTorch with Blackwell (sm_120) support has been installed."
echo ""
echo "To verify again at any time, run:"
echo "python -c \"import torch; print(torch.cuda.get_device_properties(0))\""
echo ""
echo "Verification script saved at: $WORKDIR/verify_pytorch.py"
echo ""
echo "Thank you for using the PyTorch Blackwell Installer! 🚀"