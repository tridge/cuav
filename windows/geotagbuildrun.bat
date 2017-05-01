rem This builds and runs geotag.py
rem It is useful as a quick build-n-run for debugging changes to cuav
rem This assumes Python is installed in C:\Python27
rem   If it is not, change the PYTHON_LOCATION environment variable accordingly
rem This assumes InnoSetup is installed in C:\Program Files (x86)\Inno Setup 5
rem   If it is not, change the INNOSETUP environment variable accordingly
rem This requires the MinGW compiler and libjpeg-turbo development files (libjpeg-turbo-1.5.1-gcc.exe)
rem also requires pyexiv2 - https://launchpad.net/pyexiv2
SETLOCAL enableextensions

if "%PYTHON_LOCATION%" == "" (set "PYTHON_LOCATION=C:\Python27")
if "%INNOSETUP%" == "" (set "INNOSETUP=C:\Program Files (x86)\Inno Setup 5")
if "%MINGW_LOCATION%" == "" (set "MINGW_LOCATION=C:\MinGW\bin")
if "%LIBJPEGTURBO_LOCATION%" == "" (set "LIBJPEGTURBO_LOCATION=C:\libjpeg-turbo-gcc")

rem -----Add MingW and libjpeg-turbo to path-----
SET PATH=%PATH%;%MINGW_LOCATION%;%LIBJPEGTURBO_LOCATION%\include;%LIBJPEGTURBO_LOCATION%\lib;%LIBJPEGTURBO_LOCATION%\bin

rem -----Build CUAV-----
cd ..\
"%PYTHON_LOCATION%\python" setup.py clean build --compiler=mingw32 install 

rem -----Run geotag.py-----
cd .\cuav\tools
geotag.py
pause
