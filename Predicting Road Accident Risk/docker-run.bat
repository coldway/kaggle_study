@echo off
REM Docker 自动检测 GPU/CPU 并运行训练脚本 (Windows 版本)

echo ==========================================
echo Road Accident Risk Prediction - Docker
echo 自动检测 GPU/CPU 模式
echo ==========================================
echo.

REM 检测 GPU 支持
echo 正在检测系统 GPU 支持...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] 检测到 GPU 支持
        echo.
        echo GPU 信息:
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo.
        set USE_GPU=true
        set COMPOSE_FILE=docker-compose.yml
        set DOCKERFILE=Dockerfile.gpu
        goto :gpu_detected
    )
)

echo [警告] 未检测到 GPU 支持，将使用 CPU 模式
echo.
set USE_GPU=false
set COMPOSE_FILE=docker-compose.cpu.yml
set DOCKERFILE=Dockerfile

:gpu_detected

REM 解析命令行参数
set BUILD=false
set DETACHED=false
set FORCE_CPU=false

:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--build" set BUILD=true
if "%~1"=="-b" set BUILD=true
if "%~1"=="--detached" set DETACHED=true
if "%~1"=="-d" set DETACHED=true
if "%~1"=="--cpu" (
    set FORCE_CPU=true
    set USE_GPU=false
    set COMPOSE_FILE=docker-compose.cpu.yml
    set DOCKERFILE=Dockerfile
    echo [信息] 强制使用 CPU 模式
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
shift
goto :parse_args

:end_parse

REM 检查必要的文件
if not exist "%COMPOSE_FILE%" (
    echo [错误] 找不到配置文件 %COMPOSE_FILE%
    exit /b 1
)

if not exist "%DOCKERFILE%" (
    echo [错误] 找不到 Dockerfile %DOCKERFILE%
    exit /b 1
)

REM 检查数据目录
if not exist "playground-series-s5e10" (
    echo [警告] 数据目录 playground-series-s5e10 不存在
    echo 请确保数据目录存在并包含 train.csv 和 test.csv
)

REM 创建输出目录
if not exist "output" mkdir output

REM 构建命令
set COMPOSE_CMD=docker-compose -f %COMPOSE_FILE%

if "%BUILD%"=="true" (
    echo.
    echo ==========================================
    echo 构建 Docker 镜像...
    echo ==========================================
    %COMPOSE_CMD% build --no-cache
    echo.
)

REM 运行命令
echo ==========================================
if "%USE_GPU%"=="true" (
    echo 使用模式: GPU
) else (
    echo 使用模式: CPU
)
echo 配置文件: %COMPOSE_FILE%
echo Dockerfile: %DOCKERFILE%
echo ==========================================
echo.

if "%DETACHED%"=="true" (
    echo 后台运行容器...
    %COMPOSE_CMD% up -d
    echo.
    echo 容器已在后台运行
    echo 使用以下命令查看日志:
    echo   %COMPOSE_CMD% logs -f
    echo.
    echo 使用以下命令停止容器:
    echo   %COMPOSE_CMD% down
) else (
    echo 启动容器...
    echo.
    %COMPOSE_CMD% up
)

exit /b 0

:show_help
echo 用法: %~nx0 [选项]
echo.
echo 选项:
echo   --build, -b      构建镜像后再运行
echo   --detached, -d   后台运行
echo   --cpu            强制使用 CPU 模式
echo   --help, -h       显示此帮助信息
echo.
echo 示例:
echo   %~nx0               自动检测并运行
echo   %~nx0 --build       构建并运行
echo   %~nx0 --cpu         强制使用 CPU 模式
echo   %~nx0 --detached    后台运行
exit /b 0

