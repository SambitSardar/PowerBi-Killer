@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting Excel Converter...
python excel_converter.py
pause