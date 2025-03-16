@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo DeepSeek-R1 with Logic-LLM Framework
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

:: Set API key
echo.
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
    set MODEL_NAME=deepseek_deepseek-r1:free
    echo Invalid choice. Using default: deepseek/deepseek-r1:free
)

:: Dataset split selection
echo.
echo Using test split for dataset.
set SPLIT=test

:: Number of threads
echo.
set NUM_THREADS=30
echo Using %NUM_THREADS% threads for processing.

:: Max tokens
echo.
set MAX_TOKENS=8000
echo Using max tokens: %MAX_TOKENS%

:: Backup strategy
echo.
set BACKUP_STRATEGY=LLM
echo Using backup strategy: %BACKUP_STRATEGY%

echo.
echo ===================================================
echo RUNNING DEEPSEEK WITH LOGIC-LLM FRAMEWORK
echo ===================================================
echo.

:: Run the Python script with the selected options
python3 run_deepseek_arlsat_logic_llm.py --api_key %API_KEY% --model_name "%MODEL_NAME%" --split %SPLIT% --max_new_tokens %MAX_TOKENS% --backup_strategy %BACKUP_STRATEGY% --num_threads %NUM_THREADS%

echo.
echo ===================================================
echo EXECUTION COMPLETE
echo ===================================================
echo.

pause 

:: Evaluation
:: python3 models/evaluation.py --dataset_name "AR-LSAT" --model_name "deepseek/deepseek-r1" --split test --backup "LLM"
    
:: Self-refinement
:: python3 models/self_refinement.py --model_name deepseek/deepseek-r1 --dataset_name AR-LSAT --split test --backup_strategy LLM --api_key %API_KEY%