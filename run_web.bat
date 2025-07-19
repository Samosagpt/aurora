@echo off
echo Starting Samosa GPT Web Interface...
echo Web interface will be available at: http://localhost:8501
echo Close this window to stop the application
echo.
python -m streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
pause
