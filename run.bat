@echo off
REM ===========================================================================
REM  local-rag-learning launcher
REM
REM  Usage:
REM      run.bat                     -> show CLI help
REM      run.bat ingest              -> ingest documents
REM      run.bat ingest --force      -> re-ingest everything
REM      run.bat query "..." --debug -> ask a question with debug output
REM      run.bat stats               -> show vector store stats
REM      run.bat clear               -> wipe the vector store
REM
REM  On first run this script will:
REM      1. Create a virtual environment in .venv
REM      2. Install all required Python packages
REM      3. Copy config.example.yaml to config.yaml if missing
REM  Subsequent runs reuse the existing venv and skip setup.
REM ===========================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "VENV_DIR=.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM ----- 1. Ensure Python is installed --------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [run.bat] ERROR: Python was not found on PATH.
    echo            Install Python 3.11 or newer from https://www.python.org/
    popd >nul
    exit /b 1
)

REM ----- 2. Create the virtual environment on first run ---------------------
if not exist "%PYTHON_EXE%" (
    echo [run.bat] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [run.bat] ERROR: failed to create virtual environment.
        popd >nul
        exit /b 1
    )
)

REM ----- 3. Verify required packages, install if anything is missing --------
"%PYTHON_EXE%" -c "import filelock, python_multipart, typer, chromadb, pydantic, httpx, pypdf, rich, openai, yaml, rag_app" >nul 2>&1
if errorlevel 1 (
    echo [run.bat] Installing dependencies into %VENV_DIR% ...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -e .
    if errorlevel 1 (
        echo [run.bat] ERROR: dependency installation failed.
        popd >nul
        exit /b 1
    )
)

REM ----- 4. Bootstrap config.yaml from the example on first run -------------
if not exist config.yaml (
    if exist config.example.yaml (
        echo [run.bat] Creating config.yaml from config.example.yaml ...
        copy /Y config.example.yaml config.yaml >nul
        echo [run.bat] Review config.yaml and choose your provider before querying.
    )
)

REM ----- 5. Run the CLI, forwarding every argument --------------------------
"%PYTHON_EXE%" -m rag_app %*
set "EXITCODE=%ERRORLEVEL%"

popd >nul
exit /b %EXITCODE%
