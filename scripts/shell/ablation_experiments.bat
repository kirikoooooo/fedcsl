@echo off
REM 客户端选择消融实验脚本（Windows批处理版本）
REM 使用方法: 双击运行或在命令行中执行 ablation_experiments.bat

set BASE_CONFIG=configAVG.yml
set DATASET=Epilepsy-TSTCC
set LOG_DIR=logs

REM 创建日志目录
if not exist %LOG_DIR% mkdir %LOG_DIR%

echo ========================================
echo 客户端选择消融实验
echo ========================================
echo 数据集: %DATASET%
echo 配置文件: %BASE_CONFIG%
echo 日志目录: %LOG_DIR%
echo ========================================
echo.

REM ============================================
REM Baseline: FedAvg without client selection
REM ============================================
REM echo === Baseline (FedAvg without client selection) ===
REM python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --description "Baseline_FedAvg" > %LOG_DIR%\baseline_fedavg.log 2>&1

REM ============================================
REM Ablation 1: 不同采样比例 (ratio)
REM 固定: min_prob=0.01, ema_alpha=0.2
REM ============================================
REM echo === Ablation 1: Different client selection ratios ===
REM start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.5 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.5" > %LOG_DIR%\ablation_ratio0.5.log 2>&1
REM start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.6 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.6" > %LOG_DIR%\ablation_ratio0.6.log 2>&1
REM start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.7" > %LOG_DIR%\ablation_ratio0.7.log 2>&1
REM start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.8 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ratio0.8" > %LOG_DIR%\ablation_ratio0.8.log 2>&1
REM timeout /t 5 /nobreak >nul
REM :wait1
REM tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
REM if "%ERRORLEVEL%"=="0" (
REM     timeout /t 2 /nobreak >nul
REM     goto wait1
REM )

REM ============================================
REM Ablation 2: 不同最小选择概率 (min_prob)
REM 固定: ratio=0.7, ema_alpha=0.2
REM ============================================
echo === Ablation 2: Different minimum selection probabilities ===
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.005 --ema-alpha 0.2 --description "Ablation_minprob0.005" > %LOG_DIR%\ablation_minprob0.005.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_minprob0.01" > %LOG_DIR%\ablation_minprob0.01.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.02 --ema-alpha 0.2 --description "Ablation_minprob0.02" > %LOG_DIR%\ablation_minprob0.02.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.05 --ema-alpha 0.2 --description "Ablation_minprob0.05" > %LOG_DIR%\ablation_minprob0.05.log 2>&1
timeout /t 5 /nobreak >nul
:wait2
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    timeout /t 2 /nobreak >nul
    goto wait2
)

REM ============================================
REM Ablation 3: 不同EMA平滑系数 (ema_alpha)
REM 固定: ratio=0.7, min_prob=0.01
REM ============================================
echo === Ablation 3: Different EMA smoothing coefficients ===
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.1 --description "Ablation_ema0.1" > %LOG_DIR%\ablation_ema0.1.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.2 --description "Ablation_ema0.2" > %LOG_DIR%\ablation_ema0.2.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.3 --description "Ablation_ema0.3" > %LOG_DIR%\ablation_ema0.3.log 2>&1
start /B python FedCSL_Epilepsy.py -dataset %DATASET% --config %BASE_CONFIG% --use-client-selection --client-selection-ratio 0.7 --min-selection-prob 0.01 --ema-alpha 0.5 --description "Ablation_ema0.5" > %LOG_DIR%\ablation_ema0.5.log 2>&1
timeout /t 5 /nobreak >nul
:wait3
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    timeout /t 2 /nobreak >nul
    goto wait3
)

echo.
echo ========================================
echo 所有消融实验完成！
echo ========================================
pause

