@echo off
cd /d %~dp0

set "MVS_RUNTIME_1=C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
set "MVS_RUNTIME_2=C:\Program Files\Common Files\MVS\Runtime\Win64_x64"
set "MVS_RUNTIME_3=C:\Program Files\MVS\Runtime\Win64_x64"

if exist "%MVS_RUNTIME_1%\MvCameraControl.dll" (
    set "HIKROBOT_MVS_RUNTIME=%MVS_RUNTIME_1%"
    set "PATH=%MVS_RUNTIME_1%;%PATH%"
) else if exist "%MVS_RUNTIME_2%\MvCameraControl.dll" (
    set "HIKROBOT_MVS_RUNTIME=%MVS_RUNTIME_2%"
    set "PATH=%MVS_RUNTIME_2%;%PATH%"
) else if exist "%MVS_RUNTIME_3%\MvCameraControl.dll" (
    set "HIKROBOT_MVS_RUNTIME=%MVS_RUNTIME_3%"
    set "PATH=%MVS_RUNTIME_3%;%PATH%"
)

python crack_detection_app.py
pause
