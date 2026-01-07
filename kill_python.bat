@echo off
REM 终止所有Python进程的脚本
REM 使用方法: 双击运行或在命令行中执行 kill_python.bat

echo ========================================
echo 正在终止所有Python进程...
echo ========================================

REM 列出当前运行的Python进程
echo.
echo 当前运行的Python进程:
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe"
if "%ERRORLEVEL%"=="0" (
    echo.
    echo 正在终止这些进程...
    taskkill /F /IM python.exe
    echo.
    echo ========================================
    echo 所有Python进程已终止！
    echo ========================================
) else (
    echo.
    echo 没有发现运行中的Python进程。
)

pause

