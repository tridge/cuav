rem This builds and runs geotag.py
rem It is useful as a quick build-n-run for debugging changes to cuav
rem This assumes Python is installed in C:\Python36
rem   If it is not, change the PYTHON_LOCATION environment variable accordingly

SETLOCAL enableextensions

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x86 10.0.17134.0
if "%PYTHON_LOCATION%" == "" (set "PYTHON_LOCATION=C:\Python36")

rem -----Build CUAV-----
cd ..\
"%PYTHON_LOCATION%\python" setup.py build install --user

rem -----Run geotag.py-----
cd .\cuav\tools
"%PYTHON_LOCATION%\python" geotag.py
pause
