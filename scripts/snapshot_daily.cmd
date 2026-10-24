@echo off
rem Weekday job before the US open: save the last session's CBOE chains, then run the general scanner (paper only).
cd /d "%~dp0.."
if not exist data mkdir data
if exist "data\sec_user_agent.txt" set /p SEC_USER_AGENT=<"data\sec_user_agent.txt"
".venv\Scripts\python.exe" -m omodel snapshot-options >> "data\snapshot_daily.log" 2>&1
".venv\Scripts\python.exe" -W ignore -m omodel scan >> "data\scan_daily.log" 2>&1
