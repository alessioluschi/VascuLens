#!/usr/bin/env bash
# ============================================================
# Ulcer Classification Pipeline — Environment Setup (Linux/macOS)
# Target: Python 3.10+, NVIDIA RTX 3080, CUDA 12.x
# ============================================================

set -e  # Exit on error

VENV_DIR="venv"
PYTHON_CMD=""

echo "============================================================"
echo " Ulcer Classification Pipeline - Environment Setup"
echo " Target GPU: NVIDIA RTX 3080 (10 GB VRAM)"
echo "============================================================"
echo ""

# --- Step 1: Find Python 3.10+ ---
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON_CMD="$cmd"
                echo "[INFO] Found $cmd (Python $version)"
                return 0
            fi
        fi
    done
    echo "[ERROR] Python 3.10+ not found. Please install Python 3.10 or later."
    exit 1
}

find_python

# --- Step 2: Check NVIDIA driver / CUDA availability ---
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "[WARNING] nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "[WARNING] The pipeline requires an NVIDIA GPU with CUDA support."
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
fi

# --- Step 3: Check for venv module ---
if ! "$PYTHON_CMD" -m venv --help &> /dev/null; then
    echo "[ERROR] Python venv module not available."
    echo "[INFO] On Ubuntu/Debian, install with: sudo apt install python3-venv"
    echo "[INFO] On Fedora/RHEL, install with:   sudo dnf install python3-venv"
    exit 1
fi

# --- Step 4: Create virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists at $VENV_DIR/"
    read -p "Recreate it? This will delete the existing venv. (y/n): " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        echo "[INFO] Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "[INFO] Using existing virtual environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment in $VENV_DIR/..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# --- Step 5: Activate virtual environment ---
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- Step 6: Upgrade pip ---
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# --- Step 7: Install PyTorch with CUDA (separate from requirements.txt) ---
echo "[INFO] Installing PyTorch with CUDA 12.4 support..."
if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null; then
    echo "[WARNING] CUDA 12.4 install failed. Trying CUDA 12.1..."
    if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
        echo "[ERROR] PyTorch CUDA installation failed."
        echo "[INFO] Visit https://pytorch.org/get-started/locally/ for manual install."
        exit 1
    fi
fi

# --- Step 8: Install remaining dependencies ---
echo "[INFO] Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Step 9: Verify installation ---
echo ""
echo "============================================================"
echo " Verifying installation..."
echo "============================================================"
echo ""

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('No CUDA')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU detected')"
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB') if torch.cuda.is_available() else print('No GPU detected')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import open_clip; print(f'open_clip version: {open_clip.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# --- Step 10: Check mixed precision support ---
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    supported = 'Supported' if gpu.major >= 7 else 'NOT supported — performance will be degraded'
    print(f'FP16 (mixed precision): {supported}')
else:
    print('No GPU detected — mixed precision check skipped')
"

# --- Step 11: Create output directories ---
echo "[INFO] Creating output directories..."
mkdir -p outputs/checkpoints
mkdir -p outputs/embedding_cache
mkdir -p outputs/metrics
mkdir -p outputs/heatmaps
mkdir -p outputs/normalized_cache
mkdir -p data/VENOUS
mkdir -p data/NON-VENOUS

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " To activate the environment in a new terminal:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " To run the pipeline:"
echo "   python main.py --config config.yaml --mode all"
echo ""
echo " Place your images in:"
echo "   data/VENOUS/       (venous ulcer images)"
echo "   data/NON-VENOUS/   (non-venous ulcer images)"
echo ""
echo " HuggingFace login (required for UNI model):"
echo "   huggingface-cli login"
echo "   Accept license at: https://huggingface.co/MahmoodLab/uni"
echo ""
