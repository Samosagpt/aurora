"""
Setup script for Agentic AI capabilities
Installs required dependencies for desktop control
"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies for agentic AI"""
    dependencies = [
        "pyautogui",
        "pywin32",
        "psutil",
        "mouse",
        "Pillow"
    ]
    
    print("=" * 60)
    print("AGENTIC AI - DEPENDENCY INSTALLATION")
    print("=" * 60)
    print("\nInstalling required packages for desktop control...")
    print("\nPackages to install:")
    for dep in dependencies:
        print(f"  - {dep}")
    
    print("\n" + "=" * 60)
    input("Press Enter to continue...")
    
    for dep in dependencies:
        print(f"\n📦 Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            print("   You may need to install it manually:")
            print(f"   pip install {dep}")
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE!")
    print("=" * 60)
    
    print("\n🎉 Agentic AI dependencies installed!")
    print("\nNext steps:")
    print("1. Run the test script: python test_agentic.py")
    print("2. Start Aurora: streamlit run streamlit_app.py")
    print("3. Enable Agentic Mode in the chat interface")
    print("\n📖 Read AGENTIC_GUIDE.md for usage instructions")

if __name__ == "__main__":
    install_dependencies()
