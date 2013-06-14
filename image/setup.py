from distutils.core import setup, Extension
import numpy as np
 
scanner = Extension('scanner',
                    sources = ['scanner.c'],
                    extra_compile_args=["-std=gnu99"])
 
setup (name = 'scanner',
        version = '1.0',
        description = 'CanberraUAV image scanner',
		include_dirs = [np.get_include()],
        ext_modules = [scanner])
