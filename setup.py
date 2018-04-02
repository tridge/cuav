from setuptools import setup, Extension
import numpy as np
import platform, os
import sys

version = '1.4.0'

ext_modules = []

if platform.system() == 'Windows':
    extra_compile_args=["-std=gnu99", "-O3"]
else:
    if platform.machine().find('arm') != -1:
        extra_compile_args=["-std=gnu99", "-O3", "-mfpu=neon"]
    else:
        extra_compile_args=["-std=gnu99", "-O3"]
 
scanner = Extension('cuav.image.scanner',
                    sources = ['cuav/image/scanner.c', 'cuav/image/imageutil.c'],
                    libraries = ['jpeg'],
                    extra_compile_args=extra_compile_args)
#                    extra_compile_args=extra_compile_args + ['-O0'])
ext_modules.append(scanner)
 
setup (name = 'cuav',
       zip_safe=True,
       version = version,
       description = 'CanberraUAV UAV code',
       long_description = '''A set of python libraries and tools developed by CanberraUAV for the Outback Challenge. This includes an image search algorithm with optimisation for ARM processors and a number of mission planning and analysis tools.''',
       url = 'https://github.com/CanberraUAV/cuav',
       author = 'CanberraUAV',
       install_requires = [ 'pymavlink',
                            'MAVProxy',
                            'pytest'],
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
       packages = ['cuav', 'cuav.lib', 'cuav.image', 'cuav.camera', 'cuav.uav', 'cuav.modules', 'cuav.tools'],
       scripts = [ 'cuav/tools/geosearch.py', 'cuav/tools/geotag.py',
                   'cuav/tools/cuav_lens.py', 'cuav/tools/agl_mission.py',
                   'cuav/tools/thermal_view.py'],
       package_data = { 'cuav' : [ 'data/chameleon1_arecont0.json',
                                   'image/include/*.h']},
       ext_modules = ext_modules)
