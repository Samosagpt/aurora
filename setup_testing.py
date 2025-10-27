"""
Setup script for Aurora testing infrastructure.

This script helps set up the complete testing environment including:
- Installing test dependencies
- Setting up pre-commit hooks
- Configuring pytest
- Initializing test directories
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"➤ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {description} - Success")
            if result.stdout:
                print(f"     {result.stdout.strip()}")
        else:
            print(f"  ⚠️  {description} - Warning")
            if result.stderr:
                print(f"     {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {description} - Failed")
        print(f"     Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {description} - Failed with exception")
        print(f"     Error: {e}")
        return False


def main():
    """Main setup function."""
    print_header("Aurora Testing Infrastructure Setup")

    # Get project root
    project_root = Path(__file__).parent
    print(f"📁 Project Root: {project_root}\n")

    # Step 1: Upgrade pip
    print_header("Step 1: Upgrade pip")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")

    # Step 2: Install testing dependencies
    print_header("Step 2: Install Testing Dependencies")

    test_packages = [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
        "pytest-timeout>=2.1.0",
        "pytest-mock>=3.11.0",
        "pytest-asyncio>=0.21.0",
        "coverage>=7.3.0",
    ]

    for package in test_packages:
        run_command(
            f"{sys.executable} -m pip install '{package}'",
            f"Installing {package.split('>=')[0]}",
            check=False,
        )

    # Step 3: Install code quality tools
    print_header("Step 3: Install Code Quality Tools")

    quality_packages = [
        "black>=24.1.0",
        "isort>=5.13.0",
        "flake8>=7.0.0",
        "pylint>=3.0.0",
        "mypy>=1.8.0",
        "bandit[toml]>=1.7.0",
    ]

    for package in quality_packages:
        run_command(
            f"{sys.executable} -m pip install '{package}'",
            f"Installing {package.split('>=')[0]}",
            check=False,
        )

    # Step 4: Install pre-commit
    print_header("Step 4: Install and Setup Pre-commit")

    if run_command(f"{sys.executable} -m pip install pre-commit>=3.5.0", "Installing pre-commit"):
        # Initialize pre-commit hooks
        run_command("pre-commit install", "Installing pre-commit hooks", check=False)

        # Update hooks
        run_command("pre-commit autoupdate", "Updating pre-commit hooks", check=False)

    # Step 5: Create test directories
    print_header("Step 5: Create Test Directories")

    test_dirs = [
        project_root / "tests",
        project_root / "tests" / "unit",
        project_root / "tests" / "integration",
        project_root / "tests" / "data",
    ]

    for test_dir in test_dirs:
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created directory: {test_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"  ⚠️  Could not create {test_dir}: {e}")

    # Step 6: Create __init__.py files
    print_header("Step 6: Initialize Test Packages")

    init_files = [
        project_root / "tests" / "__init__.py",
        project_root / "tests" / "unit" / "__init__.py",
        project_root / "tests" / "integration" / "__init__.py",
    ]

    for init_file in init_files:
        try:
            if not init_file.exists():
                init_file.write_text('"""Test package."""\n')
                print(f"  ✅ Created: {init_file.relative_to(project_root)}")
            else:
                print(f"  ℹ️  Already exists: {init_file.relative_to(project_root)}")
        except Exception as e:
            print(f"  ⚠️  Could not create {init_file}: {e}")

    # Step 7: Verify installation
    print_header("Step 7: Verify Installation")

    verifications = [
        ("pytest --version", "pytest"),
        ("black --version", "black"),
        ("isort --version", "isort"),
        ("flake8 --version", "flake8"),
        ("pre-commit --version", "pre-commit"),
    ]

    all_good = True
    for cmd, tool in verifications:
        if not run_command(cmd, f"Checking {tool}", check=False):
            all_good = False

    # Step 8: Run initial test
    print_header("Step 8: Run Initial Tests")

    run_command("pytest --collect-only", "Collecting tests", check=False)

    # Final summary
    print_header("Setup Complete!")

    if all_good:
        print("✅ All components installed successfully!\n")
        print("📋 Next steps:")
        print("   1. Review the test configuration in pytest.ini")
        print("   2. Check out TESTING.md for detailed documentation")
        print("   3. Run 'pytest' to execute all tests")
        print("   4. Run 'pre-commit run --all-files' to check code quality")
        print("\n🎯 Quick commands:")
        print("   • pytest              - Run all tests")
        print("   • pytest -v           - Run tests with verbose output")
        print("   • pytest -m unit      - Run only unit tests")
        print("   • pytest --cov=.      - Run tests with coverage")
        print("   • pre-commit run -a   - Run all pre-commit hooks")
    else:
        print("⚠️  Setup completed with some warnings.")
        print("   Please review the output above for any issues.")
        print("   You may need to install some dependencies manually.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
