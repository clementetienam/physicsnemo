#!/bin/bash

cat <<EOF
#############################################################
# Author: Clement Etienam (cetienam@nvidia.com)
# INSTALL NVIDIA MODULUS + PYTORCH IN A PYTHON VIRTUAL ENV
# Optimized for HPC or container environments (non-root)
#############################################################
EOF

set -e  # stop on error

echo "🚀 Creating Virtual Environment for NVIDIA Modulus Setup (Python 3 + CUDA 12.8 + H100)..."

apt update && apt install -y libx11-dev libxt6 libgl1-mesa-dev libglu1-mesa-dev libxrender1

# 🔍 Verify GPU and CUDA
echo "🔍 Verifying NVIDIA GPU and CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected!"
    exit 1
fi

# 🐍 Detect usable Python
if command -v python3.9 &> /dev/null; then
    PYTHON_BIN=python3.9
elif command -v python3.8 &> /dev/null; then
    PYTHON_BIN=python3.8
elif command -v python3 &> /dev/null; then
    PYTHON_BIN=python3
else
    echo "❌ No compatible Python (>=3.8) found!"
    exit 1
fi
echo "✅ Using Python: $($PYTHON_BIN --version)"

# 🛠 Create virtual environment
VENV_NAME="modulus_venv"
echo "🛠 Creating Python virtual environment at: $VENV_NAME"
$PYTHON_BIN -m venv $VENV_NAME
source $VENV_NAME/bin/activate

export CC=gcc
export CXX=g++
export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"

echo "✅ Python version: $(python --version)"
echo "✅ Pip version: $(pip --version)"

# 5️⃣ Clean conflicting installs
echo "🧹 Uninstalling conflicting packages..."
pip uninstall -y torch torchvision torchaudio dgl || true

# 6️⃣ Base packages
echo "📦 Installing base packages..."
pip install --upgrade pip setuptools wheel typing-extensions==4.12.2 --no-build-isolation
pip install Cython numpy --no-build-isolation
pip install torchdata


# 7️⃣ Install PyTorch 2.6.0 for CUDA 12.8
echo "🔥 Installing PyTorch 2.6.0 (cu128)..."
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  --no-build-isolation

# 8️⃣ NVIDIA Modulus and CuPy
echo "📘 Installing NVIDIA Modulus and CuPy..."
pip install nvidia-modulus==0.6.0
pip install cupy-cuda12x --no-build-isolation

# 9️⃣ DGL (choose one version!)
echo "🔁 Installing DGL (cu121)..."
#pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html --no-build-isolation
pip install vtk termcolor
# 🔟 Additional dependencies
echo "📚 Installing Python libraries..."
pip install \
    xlsxwriter PyWavelets scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown pyvista \
    gstools scikit-image accelerate loky xgboost numba scikit-learn pandas openpyxl \
    gpytorch mlflow tqdm wandb numpy pillow \
    sympy fsspec --no-build-isolation vtk mayavi

pip install hydra-core --no-build-isolation
pip install h5py
# 🔁 Final torch reinstall to ensure compatibility
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 --no-build-isolation

pip install numpy==1.24
# ✅ Verifications
echo "🧪 Running system checks..."
nvidia-smi
command -v nvcc && nvcc --version || echo "⚠️ nvcc not found"

echo "✅ PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "✅ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "✅ Python version: $(python --version)"

# 📦 Freeze environment
pip freeze > installed_modulus_env.txt

echo ""
echo "#############################################################"
echo "✅ Virtual environment setup complete!"
echo "To activate this environment, run:"
echo "source $VENV_NAME/bin/activate"
echo "#############################################################"

