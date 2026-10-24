@echo off
rem Daily CBOE chain snapshot for current S&P 500 members; scheduled after the US close.
cd /d "%~dp0.."
if not exist data mkdir data
".venv\Scripts\python.exe" -m omodel snapshot-options >> "data\snapshot_daily.log" 2>&1
