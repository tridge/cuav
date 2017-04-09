rem build the standalone Cuav tools for Windows.
rem This assumes Python is installed in C:\Python27
rem   If it is not, change the PYTHON_LOCATION environment variable accordingly
rem This assumes InnoSetup is installed in C:\Program Files (x86)\Inno Setup 5
rem   If it is not, change the INNOSETUP environment variable accordingly
rem This requires the MinGW compiler and libjpeg-turbo development files (libjpeg-turbo-1.5.1-gcc.exe)
rem This requires Pyinstaller==2.1, setuptools==19.2 and packaging==14.2
rem also requires pyexiv2 - https://launchpad.net/pyexiv2
SETLOCAL enableextensions

if "%PYTHON_LOCATION%" == "" (set "PYTHON_LOCATION=C:\Python27")
if "%INNOSETUP%" == "" (set "INNOSETUP=C:\Program Files (x86)\Inno Setup 5")

rem get the version
for /f "tokens=*" %%a in (
 '"%PYTHON_LOCATION%\python" returnVersion.py'
 ) do (
 set VERSION=%%a
 )

rem -----build the changelog-----
"%PYTHON_LOCATION%\python" createChangelog.py

rem -----Build CUAV-----
cd ..\
"%PYTHON_LOCATION%\python" setup.py clean build --compiler=mingw32 install 
"%PYTHON_LOCATION%\Scripts\pyinstaller" -y --clean .\windows\cuav.spec

rem ----Copy the files and scanner.pyd----
mkdir .\dist\cuav
xcopy .\dist\pgmconvert\* .\dist\cuav /Y /E
xcopy .\dist\geotag\* .\dist\cuav /Y /E
xcopy .\dist\geosearch\* .\dist\cuav /Y /E
xcopy .\build\lib.win32-2.7\cuav\image\scanner.pyd .\dist\cuav\cuav.image.scanner.pyd /Y

rem -----Create version Info-----
@echo off
@echo %VERSION%> .\windows\version.txt
@echo on



pause
