@echo off
echo Setting up Samosa GPT Portable...
python -m pip install --user -r requirements.txt
pip install git+https://github.com/suno-ai/bark.git
echo Setup complete!
echo.
echo To run Samosa GPT:
echo - Web version: run_web.bat
echo - Console version: run_console.bat
pause
