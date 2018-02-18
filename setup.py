from setuptools import setup, Extension
import numpy as np
import platform, os
import sys
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

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
       setup_requires=['pytest-runner'],
       tests_require=['pytest', 'pytest-mock'],
       cmdclass = {'test': PyTest},
       packages = ['cuav', 'cuav.lib', 'cuav.image', 'cuav.camera', 'cuav.uav', 'cuav.modules'],
       scripts = [ 'cuav/tools/geosearch.py', 'cuav/tools/geotag.py',
                   'cuav/tools/cuav_lens.py', 'cuav/tools/agl_mission.py',
                   'cuav/tools/thermal_view.py'],
       package_data = { 'cuav' : [ 'tests/test-8bit.pgm',
                                   'data/chameleon1_arecont0.json',
                                   'image/include/*.h']},
       ext_modules = ext_modules)
