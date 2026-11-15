@echo off
title AURASKIN - He thong Chan Doan va Tu Van Cham Soc Da AI
cd /d "%~dp0"

echo ====================================================================
echo                   AURASKIN - CLINICAL SKINCARE AI
echo        He thong Chan doan Tinh trang Da va Tu van Phac do Ca nhan
echo ====================================================================
echo.

if exist ".venv\Scripts\python.exe" goto START_SERVER

echo [i] Moi truong ao .venv chua ton tai. Dang tien hanh thiet lap...

set PY_CMD=
if exist "C:\Program Files (x86)\Thonny\python.exe" (
    set PY_CMD="C:\Program Files (x86)\Thonny\python.exe"
    goto CREATE_VENV
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    set PY_CMD=python
    goto CREATE_VENV
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    set PY_CMD=py
    goto CREATE_VENV
)

echo [X] LOI: Khong tim thay Python tren he thong.
echo Vui long cai dat Python 3.10+ tu https://www.python.org
pause
exit /b 1

:CREATE_VENV
echo [i] Dang khoi tao moi truong ao .venv...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [X] LOI: Khong the tao moi truong ao .venv.
    pause
    exit /b 1
)

echo [i] Dang cai dat cac thu vien can thiet tu requirements.txt...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

:START_SERVER
echo [OK] Moi truong Python da san sang.
echo [i] Dang khoi dong may chu tai http://localhost:8000 ...
echo [i] Trinh duyet web se tu dong mo sau giay lat.
echo.
echo Nhan to hop phim Ctrl + C de dung may chu.
echo --------------------------------------------------------------------
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [X] May chu da dung.
    pause
)
