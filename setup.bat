@echo off
echo Setting up Samosa GPT Portable...
echo.

echo Checking system compatibility...
python gpu_check.py
echo.

echo Checking PyTorch installation status...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>nul
if %errorlevel% neq 0 (
    echo PyTorch not found. Installing PyTorch...
    goto detect_and_install_pytorch
)

python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %errorlevel% equ 0 (
    echo ✓ CUDA is available and PyTorch supports it.
    goto install_requirements
) else (
    echo PyTorch found but CUDA not available. Checking hardware...
    goto detect_and_install_pytorch
)

:detect_and_install_pytorch
echo Detecting optimal PyTorch installation...

REM Check for NVIDIA GPU using nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected with NVIDIA drivers.
    
    REM Check for CUDA toolkit
    nvcc --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo CUDA toolkit found. Installing PyTorch with CUDA support...
        goto install_pytorch_cuda
    ) else (
        echo.
        echo ⚠️  NVIDIA GPU detected but CUDA toolkit not found!
        echo.
        echo Please install CUDA toolkit from:
        echo https://developer.nvidia.com/cuda-downloads
        echo.
        echo Choose one of the following options:
        echo 1. Install CUDA toolkit now and re-run this setup
        echo 2. Continue with CPU-only PyTorch (slower performance)
        echo.
        set /p choice="Enter your choice (1 or 2): "
        if "%choice%"=="1" (
            echo Opening CUDA download page...
            start https://developer.nvidia.com/cuda-downloads
            echo Please install CUDA and re-run this setup.
            pause
            exit /b 1
        ) else (
            echo Installing CPU-only PyTorch...
            goto install_pytorch_cpu
        )
    )
)

REM Check for AMD GPU using wmic
python -c "import subprocess; result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True); gpu_info = result.stdout.lower(); exit(1 if 'nvidia' in gpu_info else (2 if 'amd' in gpu_info or 'radeon' in gpu_info else 0))" 2>nul
set gpu_check=%errorlevel%

if %gpu_check% equ 1 (
    echo NVIDIA GPU detected but nvidia-smi not available.
    echo This might indicate driver issues or missing CUDA support.
    echo Installing CPU-only PyTorch...
    goto install_pytorch_cpu
) else if %gpu_check% equ 2 (
    echo AMD GPU detected. Installing PyTorch with ROCm support...
    goto install_pytorch_rocm
) else (
    echo No dedicated GPU detected. Installing CPU-only PyTorch...
    goto install_pytorch_cpu
)

:install_pytorch_cuda
echo Installing PyTorch with CUDA 12.1 support...
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
goto verify_pytorch

:install_pytorch_cpu
echo Installing CPU-only PyTorch...
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
goto verify_pytorch

:install_pytorch_rocm
echo Installing PyTorch with ROCm support...
python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
goto verify_pytorch

:verify_pytorch
echo.
echo Verifying PyTorch installation...
python -c "import torch; print('✓ PyTorch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available()); print('✓ Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Failed to verify PyTorch installation. Please check for errors above.
    pause
    exit /b 1
)

:install_requirements
echo.
echo Installing remaining requirements...
python -m pip install --user -r requirements.txt
echo Setup complete!
echo.
echo To run Samosa GPT:
echo - Web version: run_web.bat
echo - Console version: run_console.bat
echo.
pause
