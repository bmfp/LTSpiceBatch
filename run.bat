@echo off
cd "%~dp0"
set "PATH=%cd%;%PATH%"
start uv.exe --managed-python run launcher.py