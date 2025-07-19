# Samosa GPT Smart Setup Script for Windows PowerShell
# Automatically detects GPU and installs appropriate PyTorch version

Write-Host "=== Samosa GPT Smart Setup ===" -ForegroundColor Cyan
Write-Host

function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction SilentlyContinue) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Test-PythonPackage {
    param($Package)
    try {
        $result = python -c "import $Package; print('OK')" 2>$null
        return $result -eq "OK"
    } catch {
        return $false
    }
}

function Get-GPUInfo {
    Write-Host "Checking system compatibility..." -ForegroundColor Yellow
    
    # Run the detailed GPU check
    python gpu_check.py
    Write-Host
    
    # Check for NVIDIA GPU with nvidia-smi
    if (Test-Command "nvidia-smi") {
        try {
            $nvidiaOutput = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
            if ($LASTEXITCODE -eq 0 -and $nvidiaOutput) {
                Write-Host "✓ NVIDIA GPU detected with drivers" -ForegroundColor Green
                
                # Check for CUDA toolkit
                if (Test-Command "nvcc") {
                    Write-Host "✓ CUDA toolkit found" -ForegroundColor Green
                    return "nvidia-cuda"
                } else {
                    Write-Host "⚠️  NVIDIA GPU found but CUDA toolkit missing" -ForegroundColor Yellow
                    return "nvidia-no-cuda"
                }
            }
        } catch {
            # Continue to next check
        }
    }
    
    # Check for GPU using WMI
    try {
        $gpuInfo = Get-WmiObject Win32_VideoController | Select-Object Name
        $gpuNames = ($gpuInfo | ForEach-Object { $_.Name }).ToLower() -join " "
        
        if ($gpuNames -match "nvidia") {
            Write-Host "⚠️  NVIDIA GPU detected but nvidia-smi not available" -ForegroundColor Yellow
            return "nvidia-no-drivers"
        } elseif ($gpuNames -match "amd|radeon") {
            Write-Host "✓ AMD GPU detected" -ForegroundColor Green
            return "amd"
        } else {
            Write-Host "ℹ️  No dedicated GPU detected" -ForegroundColor Blue
            return "cpu"
        }
    } catch {
        Write-Host "ℹ️  Could not detect GPU, defaulting to CPU" -ForegroundColor Blue
        return "cpu"
    }
}

function Install-PyTorch {
    param($GPUType)
    
    Write-Host "Installing PyTorch..." -ForegroundColor Yellow
    
    switch ($GPUType) {
        "nvidia-cuda" {
            Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green
            python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        "nvidia-no-cuda" {
            Write-Host
            Write-Host "🚨 NVIDIA GPU detected but CUDA toolkit not found!" -ForegroundColor Red
            Write-Host
            Write-Host "To get the best performance, please install CUDA toolkit:"
            Write-Host "https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
            Write-Host
            $choice = Read-Host "Install CPU-only PyTorch for now? (y/n)"
            if ($choice -eq "y" -or $choice -eq "Y") {
                Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
                python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            } else {
                Write-Host "Opening CUDA download page..." -ForegroundColor Blue
                Start-Process "https://developer.nvidia.com/cuda-downloads"
                Write-Host "Please install CUDA and re-run this setup."
                Read-Host "Press Enter to exit"
                exit 1
            }
        }
        "nvidia-no-drivers" {
            Write-Host "Installing CPU-only PyTorch (NVIDIA drivers may need updating)..." -ForegroundColor Yellow
            python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        }
        "amd" {
            Write-Host "Installing PyTorch with ROCm support..." -ForegroundColor Green
            python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
        }
        "cpu" {
            Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Blue
            python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install PyTorch. Please check for errors above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

function Test-PyTorch {
    Write-Host
    Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
    
    try {
        $verification = python -c @"
import torch
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU Device:', torch.cuda.get_device_name(0))
    print('✓ GPU Count:', torch.cuda.device_count())
else:
    print('✓ Device: CPU')
"@
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host $verification -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "❌ Failed to verify PyTorch installation" -ForegroundColor Red
        return $false
    }
    return $false
}

# Main execution
try {
    # Check if PyTorch is already installed and working
    if (Test-PythonPackage "torch") {
        Write-Host "Checking existing PyTorch installation..." -ForegroundColor Yellow
        $torchCheck = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>$null
        
        if ($torchCheck -eq "CUDA") {
            Write-Host "✓ PyTorch with CUDA support already installed!" -ForegroundColor Green
            Write-Host "Skipping PyTorch installation..."
        } else {
            Write-Host "PyTorch found but no CUDA support. Checking hardware..." -ForegroundColor Yellow
            $gpuType = Get-GPUInfo
            
            if ($gpuType -eq "nvidia-cuda") {
                Write-Host "Upgrading to CUDA-enabled PyTorch..." -ForegroundColor Yellow
                Install-PyTorch $gpuType
                Test-PyTorch | Out-Null
            }
        }
    } else {
        # PyTorch not installed, detect and install
        $gpuType = Get-GPUInfo
        Install-PyTorch $gpuType
        
        if (-not (Test-PyTorch)) {
            exit 1
        }
    }
    
    # Install remaining requirements
    Write-Host
    Write-Host "Installing remaining requirements..." -ForegroundColor Yellow
    python -m pip install --user -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install requirements. Please check for errors above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Final verification
    Write-Host
    Write-Host "✓ Setup complete!" -ForegroundColor Green
    Write-Host
    Write-Host "=== Final Configuration ===" -ForegroundColor Cyan
    python gpu_check.py
    Write-Host
    Write-Host "To run Samosa GPT:" -ForegroundColor Cyan
    Write-Host "- Web version: run_web.bat" -ForegroundColor White
    Write-Host "- Console version: run_console.bat" -ForegroundColor White
    Write-Host
    
} catch {
    Write-Host "❌ Setup failed with error: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Read-Host "Press Enter to exit"
