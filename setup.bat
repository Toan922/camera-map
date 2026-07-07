@echo off
:: ============================================================
:: Camera Map – Setup Script
:: Run this once to install all required Python dependencies.
:: Installs PyTorch with CUDA 12.4 support (RTX 3070 / Ampere+)
:: and the Depth Anything 3 package for real-time depth inference.
:: ============================================================

echo.
echo  Camera Map – Dependency Installer
echo  ===================================

:: Check that Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python was not found on your PATH.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  and make sure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

echo.
for /f "tokens=*" %%v in ('python --version') do echo  Found: %%v

:: Bootstrap pip if missing
echo.
echo  Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  pip not found – bootstrapping...
    python -m ensurepip --upgrade
)

:: Upgrade pip silently
python -m pip install --upgrade pip --quiet

:: ── Step 1: PyTorch with CUDA 12.4 (for RTX 30xx / 40xx / 50xx GPUs) ────────
echo.
echo  [1/4] Installing PyTorch with CUDA 12.4 support...
echo  (This downloads ~2.5 GB – please be patient)
echo.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo  [WARNING] PyTorch GPU install failed. Trying CPU-only fallback...
    python -m pip install torch torchvision
)

:: ── Step 2: Core camera viewer deps ──────────────────────────────────────────
echo.
echo  [2/4] Installing core dependencies (opencv, pillow, numpy)...
python -m pip install "opencv-python>=4.8.0" "Pillow>=10.0.0" "numpy>=1.24.0,<2.0"

:: ── Step 3: Depth Anything 3 inference runtime deps ──────────────────────────
echo.
echo  [3/4] Installing Depth Anything 3 runtime dependencies...
python -m pip install "einops>=0.6.0" "huggingface_hub>=0.20" "safetensors>=0.4.0" "transformers>=4.40.0" "omegaconf"

:: ── Step 4: Depth Anything 3 package itself ──────────────────────────────────
echo.
echo  [4/4] Installing Depth Anything 3 from GitHub...
echo  (Clones the repo – requires git and internet access)
python -m pip install "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" --no-deps --ignore-requires-python
if errorlevel 1 (
    echo.
    echo  [WARNING] DA3 package install failed.
    echo  The app will fall back to the HuggingFace depth-estimation pipeline.
    echo  This still works but uses a different model (Depth-Anything-V2-Large).
)

echo.
echo  ============================================
echo   Setup complete!
echo   Run the app with:  python camera_viewer.py
echo.
echo   NOTE: On first launch, the depth model (~1.3 GB)
echo   will be downloaded from HuggingFace automatically.
echo  ============================================
echo.
pause
