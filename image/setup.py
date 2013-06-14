from distutils.core import setup, Extension
import numpy as np
import platform

if platform.system() == 'Windows':
    jpegturbo_libpath = "c:\libjpeg-turbo-gcc\lib"
    jpegturbo_incpath = "c:\libjpeg-turbo-gcc\include"
else:
    jpegturbo_libpath = "/opt/libjpeg-turbo/lib"
    jpegturbo_incpath = "/opt/libjpeg-turbo/include"
    

 
scanner = Extension('scanner',
                    sources = ['scanner.c'],
                    libraries = ['turbojpeg'],
                    library_dirs = [jpegturbo_libpath],
                    extra_compile_args=["-std=gnu99"])
 
setup (name = 'scanner',
        version = '1.0',
        description = 'CanberraUAV image scanner',
		include_dirs = [np.get_include(),
                                jpegturbo_incpath],
        ext_modules = [scanner])
