@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo DeepSeek-R1 with Logic-LLM Framework (DEBUG MODE)
echo ===================================================
echo.

:: Check if directories exist, create if not
if not exist "outputs\logic_programs" mkdir "outputs\logic_programs"
if not exist "outputs\results" mkdir "outputs\results"
if not exist "outputs\evaluations" mkdir "outputs\evaluations"

:: Load API key from .env file
echo Loading API key from .env file...
for /F "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="OPENROUTER_API_KEY" set API_KEY=%%b
)

:: Get API key
echo Using OpenRouter API key: %API_KEY:~0,8%***************

:: Model selection
echo.
echo Select DeepSeek model:
echo 1. deepseek/deepseek-r1:free (Free version)
echo 2. deepseek/deepseek-r1 (Paid version)
set /p MODEL_CHOICE=Enter choice (1-2): 

if "%MODEL_CHOICE%"=="1" (
    set MODEL_NAME=deepseek/deepseek-r1:free
) else if "%MODEL_CHOICE%"=="2" (
    set MODEL_NAME=deepseek/deepseek-r1
) else (
    set MODEL_NAME=deepseek/deepseek-r1:free
    echo Invalid choice. Using default: deepseek/deepseek-r1:free
)

:: Dataset split selection
echo.
echo Using test split for dataset.
set SPLIT=test

:: Processing method selection
echo.
echo DEBUG MODE: Using single-threaded processing for better error visibility
set PROCESSING_ARGS=

:: Max tokens
echo.
set MAX_TOKENS=1024
echo Using max tokens: %MAX_TOKENS% (reduced for debugging)

:: Backup strategy
echo.
set BACKUP_STRATEGY=random
echo Using backup strategy: %BACKUP_STRATEGY%

:: Debug options
echo.
echo Setting DEBUG environment variables for better error reporting
set PYTHONUNBUFFERED=1
set REQUESTS_CA_BUNDLE=
set REQUESTS_DEBUG=1

echo.
echo ===================================================
echo RUNNING DEEPSEEK WITH LOGIC-LLM FRAMEWORK (DEBUG)
echo ===================================================
echo.

:: First test the API connection
echo Testing API connection...
py -3 test_openrouter.py %API_KEY% %MODEL_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo API test failed. Please check your connection and API key.
    goto :end
)

echo.
echo API test successful. Continuing with main script...
echo.

:: Run the Python script with the selected options and debug settings
py -3 run_deepseek_arlsat_logic_llm.py --api_key %API_KEY% --model_name "%MODEL_NAME%" --split %SPLIT% --max_new_tokens %MAX_TOKENS% --backup_strategy %BACKUP_STRATEGY% %PROCESSING_ARGS% --verbose

:end
echo.
echo ===================================================
echo EXECUTION COMPLETE
echo ===================================================
echo.

pause 