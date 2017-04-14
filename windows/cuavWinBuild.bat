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
if "%MINGW_LOCATION%" == "" (set "MINGW_LOCATION=C:\MinGW\bin")
if "%LIBJPEGTURBO_LOCATION%" == "" (set "LIBJPEGTURBO_LOCATION=C:\libjpeg-turbo-gcc")

rem get the version
for /f "tokens=*" %%a in (
 '"%PYTHON_LOCATION%\python" returnVersion.py'
 ) do (
 set VERSION=%%a
 )

rem -----build the changelog-----
"%PYTHON_LOCATION%\python" createChangelog.py

rem -----Add MingW and libjpeg-turbo to path-----
SET PATH=%PATH%;%MINGW_LOCATION%;%LIBJPEGTURBO_LOCATION%\include;%LIBJPEGTURBO_LOCATION%\lib

rem -----Build CUAV-----
cd ..\
"%PYTHON_LOCATION%\python" setup.py clean build --compiler=mingw32 install 
"%PYTHON_LOCATION%\Scripts\pyinstaller" -y --clean .\windows\cuav.spec

rem ----Copy the files, libjpeg-turbo dll's and scanner.pyd----
mkdir .\dist\cuav
xcopy .\dist\pgmconvert\* .\dist\cuav /Y /E
xcopy .\dist\geotag\* .\dist\cuav /Y /E
xcopy .\dist\geosearch\* .\dist\cuav /Y /E
xcopy .\build\lib.win32-2.7\cuav\image\scanner.pyd .\dist\cuav\cuav.image.scanner.pyd /Y
xcopy %LIBJPEGTURBO_LOCATION%\bin\libjpeg-62.dll .\dist\cuav\ /Y
xcopy %LIBJPEGTURBO_LOCATION%\bin\libturbojpeg.dll .\dist\cuav\ /Y

rem -----Create version Info-----
@echo off
@echo %VERSION%> .\windows\version.txt
@echo on

rem -----Build the Installer-----
cd  .\windows\
rem Newer Inno Setup versions do not require a -compile flag, please add it if you have an old version
"%INNOSETUP%\ISCC.exe" /dMyAppVersion=%VERSION% cuav.iss

pause
