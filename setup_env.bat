@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM Ulcer Classification Pipeline — Environment Setup (Windows)
REM Target: Python 3.10+, NVIDIA RTX 3080, CUDA 12.x
REM ============================================================

echo ============================================================
echo  Ulcer Classification Pipeline - Environment Setup
echo  Target GPU: NVIDIA RTX 3080 (10 GB VRAM)
echo ============================================================
echo.

REM --- Step 1: Check Python version ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [INFO] Found Python %PYTHON_VERSION%

REM --- Step 2: Check NVIDIA driver / CUDA availability ---
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found. NVIDIA drivers may not be installed.
    echo [WARNING] The pipeline requires an NVIDIA GPU with CUDA support.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 1
) else (
    echo [INFO] NVIDIA driver detected:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo.
)

REM --- Step 3: Create virtual environment ---
set VENV_DIR=venv
if exist %VENV_DIR% (
    echo [INFO] Virtual environment already exists at %VENV_DIR%\
    set /p RECREATE="Recreate it? This will delete the existing venv. (y/n): "
    if /i "!RECREATE!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q %VENV_DIR%
    ) else (
        echo [INFO] Using existing virtual environment.
        goto :activate
    )
)

echo [INFO] Creating virtual environment in %VENV_DIR%\...
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:activate
REM --- Step 4: Activate virtual environment ---
echo [INFO] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM --- Step 5: Upgrade pip ---
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM --- Step 6: Install PyTorch with CUDA (separate from requirements.txt) ---
echo [INFO] Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARNING] CUDA 12.4 install failed. Trying CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERROR] PyTorch CUDA installation failed. Check your CUDA version.
        echo [INFO] Visit https://pytorch.org/get-started/locally/ for manual install.
        pause
        exit /b 1
    )
)

REM --- Step 7: Install remaining dependencies ---
echo [INFO] Installing project dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

REM --- Step 8: Verify installation ---
echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
echo.

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('No CUDA')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU')"
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB') if torch.cuda.is_available() else print('No GPU')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import open_clip; print(f'open_clip version: {open_clip.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

REM --- Step 9: Check mixed precision support ---
python -c "import torch; gpu=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(f'FP16 (mixed precision): Supported' if gpu and gpu.major >= 7 else 'FP16 (mixed precision): NOT supported') if gpu else print('No GPU detected')"

REM --- Step 10: Create output directories ---
echo [INFO] Creating output directories...
if not exist outputs\checkpoints mkdir outputs\checkpoints
if not exist outputs\embedding_cache mkdir outputs\embedding_cache
if not exist outputs\metrics mkdir outputs\metrics
if not exist outputs\heatmaps mkdir outputs\heatmaps
if not exist outputs\normalized_cache mkdir outputs\normalized_cache
if not exist data\VENOUS mkdir data\VENOUS
if not exist data\NON-VENOUS mkdir data\NON-VENOUS

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  To activate the environment in a new terminal:
echo    %VENV_DIR%\Scripts\activate.bat
echo.
echo  To run the pipeline:
echo    python main.py --config config.yaml --mode all
echo.
echo  Place your images in:
echo    data\VENOUS\       (venous ulcer images)
echo    data\NON-VENOUS\   (non-venous ulcer images)
echo.
echo  HuggingFace login (required for UNI model):
echo    huggingface-cli login
echo.
pause
