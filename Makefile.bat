@echo off
REM FusionDx -- Windows command shortcuts (use with: Makefile.bat <target>)
REM Usage: Makefile.bat test | Makefile.bat synthetic | Makefile.bat verify | etc.

SET PYTHON=venv\Scripts\python.exe
SET PYTEST=venv\Scripts\python.exe -m pytest

if "%1"=="" goto help

if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="verify" goto verify
if "%1"=="synthetic" goto synthetic
if "%1"=="test" goto test
if "%1"=="test-fast" goto test_fast
if "%1"=="test-gdc" goto test_gdc
if "%1"=="data" goto data
if "%1"=="train" goto train
if "%1"=="dashboard" goto dashboard
if "%1"=="clean" goto clean

echo Unknown target: %1
goto help

:help
echo.
echo FusionDx -- Available commands:
echo.
echo   Makefile.bat install      Install all Python dependencies
echo   Makefile.bat verify       Verify environment (GDC API + OpenSlide)
echo   Makefile.bat synthetic    Run full pipeline on synthetic data
echo   Makefile.bat test         Run all tests
echo   Makefile.bat test-fast    Run fast tests only (no model training)
echo   Makefile.bat test-gdc     Run GDC API tests (requires internet)
echo   Makefile.bat data         Run data pipeline (downloads real TCGA-BRCA data)
echo   Makefile.bat train        Train all models on real data
echo   Makefile.bat dashboard    Launch Streamlit dashboard
echo   Makefile.bat clean        Remove generated files (checkpoints, tiles)
echo.
goto end

:install
echo Installing dependencies...
venv\Scripts\pip.exe install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip.exe install -r requirements.txt
goto end

:verify
echo Verifying environment...
%PYTHON% verify_gdc.py
%PYTHON% -m src.data_pipeline --verify
goto end

:synthetic
echo Running synthetic pipeline...
%PYTHON% run_synthetic.py
goto end

:test
echo Running all tests...
%PYTEST% tests\
goto end

:test_fast
echo Running fast tests...
%PYTEST% tests\test_synthetic_data.py tests\test_evaluate.py -v
goto end

:test_gdc
echo Running GDC API tests...
%PYTEST% tests\test_gdc_api.py -v
goto end

:data
echo Running data pipeline (this will take hours and use significant disk space)...
echo Press Ctrl+C to cancel.
pause
%PYTHON% -m src.data_pipeline
goto end

:train
echo Training all models...
%PYTHON% train_all.py
goto end

:dashboard
echo Launching dashboard at http://localhost:8501 ...
venv\Scripts\streamlit.exe run app.py
goto end

:clean
echo Cleaning generated files...
echo This will remove model checkpoints and synthetic tiles.
echo Real downloaded slide data will NOT be removed.
pause
del /q results\*.pth 2>nul
del /q results\*.png 2>nul
del /q results\*.json 2>nul
del /q results\*.csv 2>nul
rmdir /s /q data\tiles\SYNTH-* 2>nul
echo Done.
goto end

:end
