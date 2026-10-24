@echo off
rem Opens the local dashboard in your browser (http://localhost:8501). Close this window to stop it.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m streamlit run app\dashboard.py --server.headless false --browser.gatherUsageStats false
