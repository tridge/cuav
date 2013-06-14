from distutils.core import setup, Extension
 
scanner = Extension('scanner',
                    sources = ['scanner.c'],
                    extra_compile_args=["-std=gnu99"])
 
setup (name = 'scanner',
        version = '1.0',
        description = 'CanberraUAV image scanner',
        ext_modules = [scanner])
