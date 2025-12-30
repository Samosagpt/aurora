#!/usr/bin/env python3
"""
GPU Detection and CUDA Compatibility Checker for Samosa GPT
"""

import platform
import subprocess
import sys


def check_nvidia_gpu():
    """Check for NVIDIA GPU using multiple methods."""
    try:
        # Method 1: nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpus = [gpu.strip() for gpu in result.stdout.strip().split("\n") if gpu.strip()]
            return gpus
    except FileNotFoundError:
        pass

    # Method 2: wmic (Windows)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                nvidia_gpus = [
                    line.strip()
                    for line in lines
                    if "nvidia" in line.lower() and line.strip() and "name" not in line.lower()
                ]
                return nvidia_gpus
        except Exception:
            pass

    return []


def check_amd_gpu():
    """Check for AMD GPU."""
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                amd_gpus = [
                    line.strip()
                    for line in lines
                    if ("amd" in line.lower() or "radeon" in line.lower())
                    and line.strip()
                    and "name" not in line.lower()
                ]
                return amd_gpus
        except Exception:
            pass
    return []


def check_pytorch_cuda():
    """Check if PyTorch is installed and CUDA-enabled."""
    try:
        import torch

        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": (
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
        }
    except ImportError:
        return {"installed": False}


def get_cuda_version():
    """Get CUDA toolkit version if available."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


def main():
    print("=== GPU and CUDA Compatibility Check ===\n")

    # Check for NVIDIA GPUs
    nvidia_gpus = check_nvidia_gpu()
    if nvidia_gpus:
        print("NVIDIA GPUs detected:")
        for gpu in nvidia_gpus:
            print(f"  - {gpu}")
    else:
        print("No NVIDIA GPUs detected")

    # Check for AMD GPUs
    amd_gpus = check_amd_gpu()
    if amd_gpus:
        print("\nAMD GPUs detected:")
        for gpu in amd_gpus:
            print(f"  - {gpu}")

    if not nvidia_gpus and not amd_gpus:
        print("No dedicated GPUs detected - will use CPU-only PyTorch")

    # Check CUDA toolkit
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"\nCUDA Toolkit: {cuda_version}")
    elif nvidia_gpus:
        print(
            "\nCUDA Toolkit: Not found (install from https://developer.nvidia.com/cuda-downloads)"
        )

    # Check PyTorch
    pytorch_info = check_pytorch_cuda()
    if pytorch_info["installed"]:
        print(f"\nPyTorch Status:")
        print(f"  Version: {pytorch_info['version']}")
        print(f"  CUDA Available: {pytorch_info['cuda_available']}")
        if pytorch_info["cuda_available"]:
            print(f"  CUDA Version: {pytorch_info.get('cuda_version', 'Unknown')}")
            print(f"  GPU Devices: {pytorch_info['device_count']}")
            for i, device in enumerate(pytorch_info["devices"]):
                print(f"    Device {i}: {device}")
    else:
        print("\nPyTorch: Not installed")

    # Recommendations
    print("\n=== Recommendations ===")
    if nvidia_gpus and not pytorch_info.get("cuda_available", False):
        if not cuda_version:
            print("1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            print("2. Reinstall PyTorch with CUDA support")
        else:
            print("Install PyTorch with CUDA support:")
            print(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
    elif amd_gpus:
        print("Install PyTorch with ROCm support:")
        print(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
        )
    elif not nvidia_gpus and not amd_gpus:
        print("Install CPU-only PyTorch:")
        print(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        )
    elif pytorch_info.get("cuda_available", False):
        print("✓ PyTorch with CUDA support is properly configured!")


if __name__ == "__main__":
    main()
