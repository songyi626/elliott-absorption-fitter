@echo off
REM ============================================================
REM launch_app.bat
REM 雙擊這個檔案 (或透過 launch_app.vbs 靜默執行) 即可開啟
REM 吸收光譜擬合工具，不需要再手動開 Anaconda Prompt。
REM
REM 使用前，請先修改下面這一行 ENV_PYTHON 的路徑，
REM 改成「你的 conda 環境裡 pythonw.exe 的實際路徑」。
REM
REM 怎麼找到這個路徑？
REM   1. 開啟 Anaconda Prompt
REM   2. 輸入: conda activate absfit
REM   3. 輸入: where pythonw
REM   4. 會印出類似這樣的路徑，把它整行複製貼到下面：
REM      C:\Users\你的帳號\anaconda3\envs\absfit\pythonw.exe
REM ============================================================

set ENV_PYTHON=C:\Users\%USERNAME%\anaconda3\envs\absfit\pythonw.exe

REM 如果上面的預設路徑找不到檔案，改用系統 PATH 中的 pythonw 當備援
if not exist "%ENV_PYTHON%" (
    set ENV_PYTHON=pythonw
)

REM 切換到本批次檔所在的資料夾 (確保能找到 main_app.py 及同目錄的模組)
cd /d "%~dp0"

start "" "%ENV_PYTHON%" main_app.py
