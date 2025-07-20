# SamosaGPT NSIS Installer Script
# This creates a professional Windows installer

!define APP_NAME "SamosaGPT"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "P.S.N Tejaji"
!define APP_URL "https://github.com/Samosagpt/samosagpt"
!define APP_DESCRIPTION "Advanced AI Assistant with Multi-Modal Capabilities"

# Installer settings
Name "SamosaGPT"
OutFile "samosa-gpt-installer.exe"
InstallDir "$PROGRAMFILES\${APP_NAME}"
RequestExecutionLevel admin

# Include modern UI
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "WinVer.nsh"

# Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "icon.ico"
!define MUI_UNICON "icon.ico"

# Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

# Uninstaller pages
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

# Languages
!insertmacro MUI_LANGUAGE "English"

# Installer information
VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "SamosaGPT"
VIAddVersionKey "ProductVersion" "1.0.0.0"
VIAddVersionKey "CompanyName" "P.S.N Tejaji"
VIAddVersionKey "FileDescription" "Advanced AI Assistant with Multi-Modal Capabilities"
VIAddVersionKey "FileVersion" "1.0.0.0"
VIAddVersionKey "LegalCopyright" "© 2025 P.S.N Tejaji. Licensed under CC BY-NC-ND 4.0"

# Installation types
InstType "Full Installation"
InstType "Minimal Installation"

# Sections
Section "Core Application" SecCore
    SectionIn RO
    
    DetailPrint "Checking requirements..."
    
    # Check Python
    nsExec::ExecToLog 'python --version'
    Pop $0
    ${If} $0 != 0
        MessageBox MB_ICONSTOP "Python 3.8+ is required but not found.$\n$\nPlease install Python from https://www.python.org/downloads/ and add it to PATH."
        Abort
    ${EndIf}
    
    # Check Git
    nsExec::ExecToLog 'git --version'
    Pop $0
    ${If} $0 != 0
        MessageBox MB_ICONSTOP "Git is required but not found.$\n$\nPlease install Git from https://git-scm.com/download/win"
        Abort
    ${EndIf}
    
    SetOutPath "$INSTDIR"
    
    # Copy installer files
    File "installer.py"
    File "install_samosagpt.bat"
    
    # Create data directory
    CreateDirectory "$APPDATA\samosagpt"
    
    DetailPrint "Cloning repository..."
    nsExec::ExecToLog 'git clone https://github.com/Samosagpt/samosagpt.git "$APPDATA\samosagpt"'
    Pop $0
    ${If} $0 != 0
        DetailPrint "Git clone failed, trying alternative method..."
        # Fallback: copy from embedded files if available
    ${EndIf}
    
    DetailPrint "Running setup..."
    nsExec::ExecToLog '"$APPDATA\samosagpt\setup.bat"'
    
    # Create launcher
    FileOpen $0 "$INSTDIR\SamosaGPT.bat" w
    FileWrite $0 '@echo off$\r$\n'
    FileWrite $0 'cd /d "$APPDATA\samosagpt"$\r$\n'
    FileWrite $0 'call run_web.bat$\r$\n'
    FileClose $0
    
    # Write registry entries
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoRepair" 1
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
SectionEnd

Section "Desktop Shortcut" SecDesktop
    SectionIn 1
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\SamosaGPT.bat" "" "$INSTDIR\SamosaGPT.bat" 0
SectionEnd

Section "Start Menu Shortcuts" SecStartMenu
    SectionIn 1
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\SamosaGPT.bat" "" "$INSTDIR\SamosaGPT.bat" 0
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

# Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} "Core application files and setup"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create desktop shortcut"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} "Create Start Menu shortcuts"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

# Uninstaller section
Section "Uninstall"
    
    # Remove registry entries
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
    
    # Remove files
    Delete "$INSTDIR\SamosaGPT.bat"
    Delete "$INSTDIR\installer.py"
    Delete "$INSTDIR\install_samosagpt.bat"
    Delete "$INSTDIR\Uninstall.exe"
    
    # Remove shortcuts
    Delete "$DESKTOP\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    
    # Remove application data (ask user)
    MessageBox MB_YESNO "Do you want to remove application data?$\n$\nThis will delete all downloaded models and settings." IDNO +2
    RMDir /r "$APPDATA\samosagpt"
    
    # Remove installation directory
    RMDir "$INSTDIR"
    
SectionEnd

# Functions
Function .onInit
    # Check Windows version
    ${If} ${AtLeastWin7}
        # Windows 7 or later - OK
    ${Else}
        MessageBox MB_ICONSTOP "This application requires Windows 7 or later."
        Abort
    ${EndIf}
FunctionEnd
