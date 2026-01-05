@echo off
cd /d "C:\Users\User\hospital_final"

REM 明確使用「Miniconda」的 activate.bat 來啟用 hospital
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" hospital

python VF_center.py
pause
