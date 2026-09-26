@echo off
REM ===========================================================================
REM  local-rag-learning GUI launcher
REM
REM  Same auto-setup as run.bat (venv, deps, config.yaml bootstrap) but
REM  opens the PySide6 desktop GUI instead of the CLI.
REM
REM  Usage:
REM      gui.bat
REM ===========================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\pythonw.exe"
set "PYTHON_CONSOLE_EXE=%VENV_DIR%\Scripts\python.exe"

REM ----- 1. Ensure Python is installed --------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [gui.bat] ERROR: Python was not found on PATH.
    echo            Install Python 3.11 or newer from https://www.python.org/
    popd >nul
    exit /b 1
)

REM ----- 2. Create the virtual environment on first run ---------------------
if not exist "%PYTHON_CONSOLE_EXE%" (
    echo [gui.bat] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [gui.bat] ERROR: failed to create virtual environment.
        popd >nul
        exit /b 1
    )
)

REM ----- 3. Verify required packages, install if anything is missing --------
"%PYTHON_CONSOLE_EXE%" -c "import filelock, python_multipart, typer, chromadb, pydantic, httpx, pypdf, rich, openai, yaml, PySide6, rag_app" >nul 2>&1
if errorlevel 1 (
    echo [gui.bat] Installing dependencies into %VENV_DIR% ...
    "%PYTHON_CONSOLE_EXE%" -m pip install --upgrade pip
    "%PYTHON_CONSOLE_EXE%" -m pip install -e ".[gui]"
    if errorlevel 1 (
        echo [gui.bat] ERROR: dependency installation failed.
        popd >nul
        exit /b 1
    )
)

REM ----- 4. Bootstrap config.yaml from the example on first run -------------
if not exist config.yaml (
    if exist config.example.yaml (
        echo [gui.bat] Creating config.yaml from config.example.yaml ...
        copy /Y config.example.yaml config.yaml >nul
    )
)

REM ----- 5. Launch the GUI --------------------------------------------------
REM  pythonw.exe runs without a console window. Fall back to python.exe if
REM  pythonw.exe doesn't exist in the venv.
if exist "%PYTHON_EXE%" (
    start "" "%PYTHON_EXE%" -m rag_app.gui
) else (
    "%PYTHON_CONSOLE_EXE%" -m rag_app.gui
)

popd >nul
exit /b 0
