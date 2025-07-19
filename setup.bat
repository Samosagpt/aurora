@echo off
echo Setting up Samosa GPT Portable...
python -m pip install --user -r requirements.txt
cd streamlit_markdown_select
cd frontend
npm install
npm run build
echo Setup complete!
echo.
echo To run Samosa GPT:
echo - Web version: run_web.bat
echo - Console version: run_console.bat
pause
