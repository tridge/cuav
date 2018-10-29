rem build the standalone Cuav tools for Windows.
rem This assumes Python is installed in C:\Python36
rem   If it is not, change the PYTHON_LOCATION environment variable accordingly
rem This assumes InnoSetup is installed in C:\Program Files (x86)\Inno Setup 5

SETLOCAL enableextensions

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86 10.0.17134.0
if "%PYTHON_LOCATION%" == "" (set "PYTHON_LOCATION=C:\Python36")
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
"%PYTHON_LOCATION%\python" setup.py build install --user
"%PYTHON_LOCATION%\Scripts\pyinstaller" -y --clean .\windows\cuav.spec

rem ----Copy the files----
mkdir .\dist\cuav
xcopy .\dist\geotag\* .\dist\cuav /Y /E
xcopy .\dist\geosearch\* .\dist\cuav /Y /E


rem -----Create version Info-----
@echo off
@echo %VERSION%> .\windows\version.txt
@echo on

rem -----Build the Installer-----
cd  .\windows\
rem Newer Inno Setup versions do not require a -compile flag, please add it if you have an old version
"%INNOSETUP%\ISCC.exe" /dMyAppVersion=%VERSION% cuav.iss

pause
