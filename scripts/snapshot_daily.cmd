@echo off
rem Weekday job after the US close: save today's CBOE chains, then score the shortlist (paper only).
cd /d "%~dp0.."
if not exist data mkdir data
".venv\Scripts\python.exe" -m omodel snapshot-options >> "data\snapshot_daily.log" 2>&1
".venv\Scripts\python.exe" -m omodel shortlist >> "data\shortlist_daily.log" 2>&1
