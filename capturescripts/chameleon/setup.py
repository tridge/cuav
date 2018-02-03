#from distutils.core import setup, Extension
import platform, os
from setuptools import setup, Extension
import numpy as np

ext_modules = []

if platform.system() == 'Windows':
    extra_compile_args=["-std=gnu99", "-O3"]
else:
    if platform.machine().find('arm') != -1:
        extra_compile_args=["-std=gnu99", "-O3", "-mfpu=neon"]
    else:
        extra_compile_args=["-std=gnu99", "-O3"]

    chameleon = Extension('chameleon',
                          sources = ['chameleon_py.c',
                                     'chameleon.c',
                                     'chameleon_util.c'],
                          libraries = ['dc1394', 'm', 'usb-1.0'],
                          extra_compile_args=extra_compile_args + ['-O0'])
    ext_modules.append(chameleon)
    
setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include(), './include'],
)
