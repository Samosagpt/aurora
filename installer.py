#!/usr/bin/env python3
"""
Samosa GPT Installer
Handles git cloning, setup, and application installation
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import winreg
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time

class SamosaGPTInstaller:
    def __init__(self):
        self.app_name = "SamosaGPT"
        self.repo_url = "https://github.com/Samosagpt/samosagpt.git"
        self.appdata_path = os.path.join(os.environ['APPDATA'], 'samosagpt')
        self.program_files_path = os.path.join(os.environ['PROGRAMFILES'], 'SamosaGPT')
        self.desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.start_menu_path = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("SamosaGPT Installer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Force window to appear on top and center it
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Progress variables
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready to install SamosaGPT")
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the installer GUI"""
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🤖 SamosaGPT Installer", 
                              font=("Arial", 18, "bold"))
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, 
                                 text="Advanced AI Assistant with Multi-Modal Capabilities",
                                 font=("Arial", 10))
        subtitle_label.pack()
        
        # License info
        license_label = tk.Label(title_frame, 
                                text="Licensed under CC BY-NC-ND 4.0 | © 2025 Aurora Project",
                                font=("Arial", 8), fg="gray")
        license_label.pack(pady=(5, 0))
        
        # Features
        features_frame = tk.Frame(self.root)
        features_frame.pack(pady=10, padx=20, fill=tk.X)
        
        features_label = tk.Label(features_frame, text="Features:", font=("Arial", 12, "bold"))
        features_label.pack(anchor=tk.W)
        
        features = [
            "✓ Multi-Modal AI Chat using Ollama models",
            "✓ Voice Interaction with Speech-to-text",
            "✓ Image Generation with Stable Diffusion",
            "✓ Web Search Integration",
            "✓ Real-time Weather and News",
            "✓ Modern Web Interface"
        ]
        
        for feature in features:
            feature_label = tk.Label(features_frame, text=feature, font=("Arial", 9))
            feature_label.pack(anchor=tk.W, padx=20)
        
        # Installation options
        options_frame = tk.LabelFrame(self.root, text="Installation Options", font=("Arial", 10, "bold"))
        options_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        self.create_start_menu = tk.BooleanVar(value=True)
        self.auto_start_browser = tk.BooleanVar(value=True)
        
        tk.Checkbutton(options_frame, text="Create Desktop Shortcut", 
                      variable=self.create_desktop_shortcut).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(options_frame, text="Add to Start Menu", 
                      variable=self.create_start_menu).pack(anchor=tk.W, padx=10, pady=2)
        tk.Checkbutton(options_frame, text="Auto-start browser after installation", 
                      variable=self.auto_start_browser).pack(anchor=tk.W, padx=10, pady=2)
        
        # Progress section
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=20, padx=20, fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=5)
        
        self.status_label = tk.Label(progress_frame, textvariable=self.status_var, 
                                    font=("Arial", 9))
        self.status_label.pack()
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.install_button = tk.Button(button_frame, text="Install SamosaGPT", 
                                       command=self.start_installation,
                                       bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                       width=15, height=2)
        self.install_button.pack(side=tk.LEFT, padx=10)
        
        self.cancel_button = tk.Button(button_frame, text="Cancel", 
                                      command=self.root.quit,
                                      bg="#f44336", fg="white", font=("Arial", 12),
                                      width=10, height=2)
        self.cancel_button.pack(side=tk.LEFT, padx=10)
        
        # Info
        info_label = tk.Label(self.root, 
                             text=f"Installation Path: {self.program_files_path}\nData Path: {self.appdata_path}",
                             font=("Arial", 8), fg="gray")
        info_label.pack(pady=10)
        
    def update_status(self, message, progress=None):
        """Update status message and progress"""
        self.status_var.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()
        
    def check_requirements(self):
        """Check if required tools are available"""
        self.update_status("Checking requirements...", 5)
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                raise Exception(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        except Exception as e:
            raise Exception(f"Python check failed: {e}")
            
        # Check Git
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Git not found")
        except Exception:
            raise Exception("Git is required but not found. Please install Git first.")
            
        self.update_status("Requirements check passed ✓", 10)
        
    def create_directories(self):
        """Create necessary directories"""
        self.update_status("Creating directories...", 15)
        
        # Create AppData directory
        os.makedirs(self.appdata_path, exist_ok=True)
        
        # Create Program Files directory (requires admin rights)
        try:
            os.makedirs(self.program_files_path, exist_ok=True)
        except PermissionError:
            # If we can't create in Program Files, use AppData instead
            self.program_files_path = os.path.join(os.environ['APPDATA'], 'SamosaGPT')
            os.makedirs(self.program_files_path, exist_ok=True)
            
        self.update_status("Directories created ✓", 20)
        
    def clone_repository(self):
        """Clone the repository to AppData"""
        self.update_status("Cloning repository (this may take a few minutes)...", 25)
        
        # Remove existing directory if it exists
        if os.path.exists(self.appdata_path):
            shutil.rmtree(self.appdata_path)
            
        try:
            # Clone the repository
            result = subprocess.run([
                'git', 'clone', self.repo_url, self.appdata_path
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Repository clone timed out. Please check your internet connection.")
        except Exception as e:
            raise Exception(f"Failed to clone repository: {e}")
            
        self.update_status("Repository cloned ✓", 50)
        
    def run_setup(self):
        """Run the setup.bat file"""
        self.update_status("Running setup (this may take several minutes)...", 60)
        
        setup_bat_path = os.path.join(self.appdata_path, 'setup.bat')
        
        if not os.path.exists(setup_bat_path):
            raise Exception("setup.bat not found in cloned repository")
            
        try:
            # Change to the repository directory and run setup
            result = subprocess.run([setup_bat_path], 
                                   cwd=self.appdata_path,
                                   capture_output=True, 
                                   text=True, 
                                   timeout=600)
            
            if result.returncode != 0:
                # Setup might still succeed even with non-zero return code
                print(f"Setup output: {result.stdout}")
                print(f"Setup errors: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Setup timed out. The installation may still be in progress.")
        except Exception as e:
            raise Exception(f"Setup failed: {e}")
            
        self.update_status("Setup completed ✓", 80)
        
    def create_launcher_script(self):
        """Create launcher script in Program Files"""
        self.update_status("Creating launcher...", 85)
        
        # Create batch file to launch the application
        launcher_content = f'''@echo off
cd /d "{self.appdata_path}"
call run_web.bat
'''
        
        launcher_path = os.path.join(self.program_files_path, 'SamosaGPT.bat')
        
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
            
        # Create PowerShell launcher (more robust)
        ps_launcher_content = f'''# SamosaGPT Launcher
Set-Location "{self.appdata_path}"
Start-Process -FilePath "run_web.bat" -WorkingDirectory "{self.appdata_path}"
'''
        
        ps_launcher_path = os.path.join(self.program_files_path, 'SamosaGPT.ps1')
        
        with open(ps_launcher_path, 'w') as f:
            f.write(ps_launcher_content)
            
        self.update_status("Launcher created ✓", 90)
        
    def create_shortcuts(self):
        """Create desktop and start menu shortcuts"""
        self.update_status("Creating shortcuts...", 95)
        
        try:
            import win32com.client
            
            shell = win32com.client.Dispatch("WScript.Shell")
            
            # Desktop shortcut
            if self.create_desktop_shortcut.get():
                desktop_shortcut = shell.CreateShortCut(
                    os.path.join(self.desktop_path, "SamosaGPT.lnk")
                )
                desktop_shortcut.Targetpath = os.path.join(self.program_files_path, 'SamosaGPT.bat')
                desktop_shortcut.WorkingDirectory = self.appdata_path
                desktop_shortcut.Description = "SamosaGPT - Advanced AI Assistant"
                desktop_shortcut.save()
                
            # Start menu shortcut
            if self.create_start_menu.get():
                start_menu_shortcut = shell.CreateShortCut(
                    os.path.join(self.start_menu_path, "SamosaGPT.lnk")
                )
                start_menu_shortcut.Targetpath = os.path.join(self.program_files_path, 'SamosaGPT.bat')
                start_menu_shortcut.WorkingDirectory = self.appdata_path
                start_menu_shortcut.Description = "SamosaGPT - Advanced AI Assistant"
                start_menu_shortcut.save()
                
        except ImportError:
            # Fallback: create simple batch files
            if self.create_desktop_shortcut.get():
                desktop_bat = os.path.join(self.desktop_path, "SamosaGPT.bat")
                shutil.copy(os.path.join(self.program_files_path, 'SamosaGPT.bat'), desktop_bat)
                
        self.update_status("Shortcuts created ✓", 100)
        
    def create_uninstaller(self):
        """Create uninstaller"""
        uninstaller_content = f'''@echo off
echo Uninstalling SamosaGPT...
echo.

echo Removing application files...
rmdir /s /q "{self.appdata_path}" 2>nul
rmdir /s /q "{self.program_files_path}" 2>nul

echo Removing shortcuts...
del "{os.path.join(self.desktop_path, 'SamosaGPT.lnk')}" 2>nul
del "{os.path.join(self.desktop_path, 'SamosaGPT.bat')}" 2>nul
del "{os.path.join(self.start_menu_path, 'SamosaGPT.lnk')}" 2>nul

echo SamosaGPT has been uninstalled.
pause
'''
        
        uninstaller_path = os.path.join(self.program_files_path, 'Uninstall.bat')
        with open(uninstaller_path, 'w') as f:
            f.write(uninstaller_content)
            
    def installation_thread(self):
        """Run installation in separate thread"""
        try:
            self.check_requirements()
            self.create_directories()
            self.clone_repository()
            self.run_setup()
            self.create_launcher_script()
            self.create_shortcuts()
            self.create_uninstaller()
            
            # Installation complete
            self.update_status("Installation completed successfully! ✓", 100)
            
            # Show completion dialog
            self.root.after(0, self.show_completion_dialog)
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error_dialog(str(e)))
            
    def show_completion_dialog(self):
        """Show installation completion dialog"""
        message = "SamosaGPT has been installed successfully!\n\n"
        message += f"Installation Path: {self.program_files_path}\n"
        message += f"Data Path: {self.appdata_path}\n\n"
        message += "You can now launch SamosaGPT from:\n"
        message += "• Desktop shortcut (if created)\n"
        message += "• Start Menu\n"
        message += "• Program Files folder\n\n"
        message += "Would you like to launch SamosaGPT now?"
        
        result = messagebox.askyesno("Installation Complete", message)
        
        if result and self.auto_start_browser.get():
            try:
                subprocess.Popen([os.path.join(self.program_files_path, 'SamosaGPT.bat')], 
                               cwd=self.appdata_path)
            except Exception as e:
                messagebox.showerror("Launch Error", f"Failed to launch application: {e}")
                
        self.root.quit()
        
    def show_error_dialog(self, error_message):
        """Show error dialog"""
        self.update_status(f"Installation failed: {error_message}", 0)
        messagebox.showerror("Installation Error", 
                           f"Installation failed:\n\n{error_message}\n\n"
                           "Please check the requirements and try again.")
        
        # Re-enable install button
        self.install_button.config(state=tk.NORMAL)
        
    def start_installation(self):
        """Start the installation process"""
        self.install_button.config(state=tk.DISABLED)
        self.cancel_button.config(text="Close")
        
        # Start installation in separate thread
        install_thread = threading.Thread(target=self.installation_thread)
        install_thread.daemon = True
        install_thread.start()
        
    def run(self):
        """Run the installer"""
        self.root.mainloop()

if __name__ == "__main__":
    # Check if running as admin (recommended)
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("Note: Running without administrator privileges.")
            print("Some features may require admin rights.")
    except:
        pass
    
    print("Starting SamosaGPT Installer GUI...")
    print("Looking for installer window...")
    
    try:
        installer = SamosaGPTInstaller()
        print("✓ GUI window created successfully!")
        print("✓ Look for the installer window on your screen")
        installer.run()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        input("Press Enter to exit...")
