# Installation Guide

## Prerequisites
- Python 3.8 or higher
- Git
- Ollama (for chat models)
- GPU (optional but recommended)
- Internet connection

## Step 1: Clone the Repository
```bash
git clone https://github.com/Tejaji-0/aurora.git
cd aurora
```

## Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```
## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
## Step 4: Configure Environment
```bash
Edit .env and config.yaml as described in CONFIG.md
```
## Step 5: Run the Application
``` bash
python app.py
```
## Step 6: Access AURORA
```bash 
Open your browser and go to:
http://localhost:8080
```
## Optional: Docker Setup
```bash
docker build -t aurora .
docker run -p 8080:8080 aurora
```
## Troubleshooting
- Ensure Ollama is running (ollama serve)
- Check Python version compatibility
- Review logs in /logs