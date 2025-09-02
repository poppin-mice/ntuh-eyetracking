@echo off
cd /d "C:\Users\User\hospital_final"

REM 明確使用「Miniconda」的 activate.bat 來啟用 hospital
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" hospital

python VA_center.py
pause
