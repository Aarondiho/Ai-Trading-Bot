"""
Dependency Installer for Platform Mind Reader
Automatically installs required packages
"""

import subprocess
import sys
import importlib

def check_and_install(package, pip_name=None):
    """Check if package is installed, install if not"""
    pip_name = pip_name or package
    
    try:
        importlib.import_module(package)
        print(f"✅ {package} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {package} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False

def main():
    """Install all required dependencies"""
    print("🚀 Platform Mind Reader - Dependency Installer")
    print("="*50)
    
    dependencies = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("tensorflow", "tensorflow"),
        ("talib", "talib-binary"),
        ("websockets", "websockets"),
        ("requests", "requests"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("ruptures", "ruptures"),
        ("markovify", "markovify"),
    ]
    
    print("Checking and installing dependencies...")
    print("-" * 30)
    
    results = []
    for package, pip_name in dependencies:
        success = check_and_install(package, pip_name)
        results.append((package, success))
    
    print("\n" + "="*50)
    print("INSTALLATION SUMMARY:")
    print("="*50)
    
    successful = [p for p, s in results if s]
    failed = [p for p, s in results if not s]
    
    if successful:
        print(f"✅ Successfully installed: {', '.join(successful)}")
    
    if failed:
        print(f"❌ Failed to install: {', '.join(failed)}")
        print("\nPlease install these packages manually:")
        for package in failed:
            print(f"  pip install {package}")
    
    if not failed:
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run: python run.py")
    else:
        print("\n⚠️ Some dependencies failed to install.")
        print("Please check the errors above and install manually.")

if __name__ == "__main__":
    main()
