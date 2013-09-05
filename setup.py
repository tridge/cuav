from distutils.core import setup, Extension
import numpy as np
import platform

version = '1.0.8'

ext_modules = []

if platform.system() == 'Windows':
    jpegturbo_libpath = "c:\libjpeg-turbo-gcc\lib"
    jpegturbo_incpath = "c:\libjpeg-turbo-gcc\include"
    extra_compile_args=["-std=gnu99", "-O3"]
else:
    jpegturbo_libpath = "/opt/libjpeg-turbo/lib"
    jpegturbo_incpath = "/opt/libjpeg-turbo/include"
    if platform.machine().find('arm') != -1:
        extra_compile_args=["-std=gnu99", "-O3", "-mfpu=neon"]
    else:
        extra_compile_args=["-std=gnu99", "-O3"]

    chameleon = Extension('cuav.camera.chameleon',
                          sources = ['camera/chameleon_py.c', 'camera/chameleon.c', 'camera/chameleon_util.c'],
                          libraries = ['dc1394', 'm', 'usb-1.0'],
                          extra_compile_args=extra_compile_args + ['-O0'])
    ext_modules.append(chameleon)

 
scanner = Extension('cuav.image.scanner',
                    sources = ['image/scanner.c', 'image/imageutil.c'],
                    libraries = ['turbojpeg'],
                    library_dirs = [jpegturbo_libpath],
                    extra_compile_args=extra_compile_args)
#                    extra_compile_args=extra_compile_args + ['-O0'])
ext_modules.append(scanner)
 
setup (name = 'cuav',
       version = version,
       description = 'CanberraUAV UAV code',
       long_description = '''A set of python libraries and tools developed by CanberraUAV for the Outback Challenge. This includes an image search algorithm with optimisation for ARM processors and a number of mission planning and analysis tools.''',
       url = 'https://github.com/CanberraUAV/cuav',
       author = 'CanberraUAV',
       requires = [ 'pymavlink (>=1.1.2)' ],
       author_email = 'andrew-cuav@tridgell.net',
       classifiers=['Development Status :: 4 - Beta',
                    'Environment :: Console',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                    'Operating System :: OS Independent',
                    'Programming Language :: Python :: 2.7',
                    'Topic :: Scientific/Engineering'
                    ],
       license='GPLv3',
       include_dirs = [np.get_include(),
                       jpegturbo_incpath],
       package_dir = { 'cuav' : '.' },
       packages = ['cuav', 'cuav.lib', 'cuav.image', 'cuav.camera', 'cuav.uav'],
       scripts = [ 'tools/geosearch.py', 'tools/geotag.py',
                   'tools/cuav_lens.py', 'tools/agl_mission.py',
                   'tools/pgm_convert.py',
                   'tests/cuav_benchmark.py' ],
       package_data = { 'cuav' : [ 'tests/test-8bit.pgm', 'data/chameleon1_arecont0.json' ]},
       ext_modules = ext_modules)
