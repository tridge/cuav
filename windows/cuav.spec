# -*- mode: python -*-
# spec file for pyinstaller to build cuav for windows

#Gooey_languages and gooey_images are used to fetch the files and solve the problem that was occuring preivously. (i.e : Language file not found)
gooey_languages = Tree('C:/Python27/Lib/site-packages/gooey/languages', prefix = 'gooey/languages')
gooey_images = Tree('C:/Python27/Lib/site-packages/gooey/images', prefix = 'gooey/images')

geotagAny = Analysis(['.\\cuav\\tools\\geotag.py'],
             pathex=[os.path.abspath('.')],
             # for some unknown reason these hidden imports don't pull in
             # all the needed pieces, so we also import them in mavproxy.py
             hiddenimports=['UserList', 'UserString',
                            'pymavlink.mavwp', 'pymavlink.mavutil', 'pymavlink.dialects.v20.ardupilotmega',
                            'pymavlink.dialects.v10.ardupilotmega',
                            'pymavlink.dialects.v20.common', 'pymavlink.dialects.v10.common',
                            'pymavlink.dialects.v20.ASLUAV', 'pymavlink.dialects.v10.ASLUAV',
                            'pymavlink.dialects.v20.autoquad', 'pymavlink.dialects.v10.autoquad',
                            'pymavlink.dialects.v20.matrixpilot', 'pymavlink.dialects.v10.matrixpilot',
                            'pymavlink.dialects.v20.minimal', 'pymavlink.dialects.v10.minimal',
                            'pymavlink.dialects.v20.paparazzi', 'pymavlink.dialects.v10.paparazzi',
                            'pymavlink.dialects.v20.slugs', 'pymavlink.dialects.v10.slugs',
                            'pymavlink.dialects.v20.standard', 'pymavlink.dialects.v10.standard',
                            'pymavlink.dialects.v20.ualberta', 'pymavlink.dialects.v10.ualberta',
                            'pymavlink.dialects.v20.uAvionix', 'pymavlink.dialects.v10.uAvionix', 'gooey'],
             excludes=['tcl', 'tk', 'Tkinter', 'tkinter', '_tkinter'],
             hookspath=None,
             runtime_hooks=None)
pgmconvertAny = Analysis(['.\\cuav\\tools\\pgm_convert.py'],
             pathex=[os.path.abspath('.')],
             # for some unknown reason these hidden imports don't pull in
             # all the needed pieces, so we also import them in mavproxy.py
             hiddenimports=['UserList', 'UserString', 'gooey'],
             excludes=['tcl', 'tk', 'Tkinter', 'tkinter', '_tkinter'],
             hookspath=None,
             runtime_hooks=None)
geosearchAny = Analysis(['.\\cuav\\tools\\geosearch.py'],
             pathex=[os.path.abspath('.')],
             # for some unknown reason these hidden imports don't pull in
             # all the needed pieces, so we also import them in mavproxy.py
             hiddenimports=['UserList', 'UserString',
                            'pymavlink.mavwp', 'pymavlink.mavutil', 'pymavlink.dialects.v20.ardupilotmega',
                            'pymavlink.dialects.v10.ardupilotmega',
                            'pymavlink.dialects.v20.common', 'pymavlink.dialects.v10.common',
                            'pymavlink.dialects.v20.ASLUAV', 'pymavlink.dialects.v10.ASLUAV',
                            'pymavlink.dialects.v20.autoquad', 'pymavlink.dialects.v10.autoquad',
                            'pymavlink.dialects.v20.matrixpilot', 'pymavlink.dialects.v10.matrixpilot',
                            'pymavlink.dialects.v20.minimal', 'pymavlink.dialects.v10.minimal',
                            'pymavlink.dialects.v20.paparazzi', 'pymavlink.dialects.v10.paparazzi',
                            'pymavlink.dialects.v20.slugs', 'pymavlink.dialects.v10.slugs',
                            'pymavlink.dialects.v20.standard', 'pymavlink.dialects.v10.standard',
                            'pymavlink.dialects.v20.ualberta', 'pymavlink.dialects.v10.ualberta',
                            'pymavlink.dialects.v20.uAvionix', 'pymavlink.dialects.v10.uAvionix', 'gooey'],
             excludes=[],
             hookspath=None,
             runtime_hooks=None)
             
MERGE( (geotagAny, 'geotag', 'geotag'), (pgmconvertAny, 'pgmconvert', 'pgmconvert'), (geosearchAny, 'geosearch', 'geosearch') )

geotag_pyz = PYZ(geotagAny.pure)
geotag_exe = EXE(geotag_pyz,
          geotagAny.scripts,
          exclude_binaries=True,
          name='geotag.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
geotag_coll = COLLECT(geotag_exe,
               geotagAny.binaries,
               geotagAny.zipfiles,
               geotagAny.datas,
               gooey_languages,
               gooey_images,
               strip=None,
               upx=True,
               name='geotag')

pgmconvert_pyz = PYZ(pgmconvertAny.pure)
pgmconvert_exe = EXE(pgmconvert_pyz,
          pgmconvertAny.scripts,
          exclude_binaries=True,
          name='pgmconvert.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
pgmconvert_coll = COLLECT(pgmconvert_exe,
               pgmconvertAny.binaries,
               pgmconvertAny.zipfiles,
               pgmconvertAny.datas,
               gooey_languages,
               gooey_images,
               strip=None,
               upx=True,
               name='pgmconvert')

geosearch_pyz = PYZ(geosearchAny.pure)
geosearch_exe = EXE(geosearch_pyz,
          geosearchAny.scripts,
          exclude_binaries=True,
          name='geosearch.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
geosearch_coll = COLLECT(geosearch_exe,
               geosearchAny.binaries,
               geosearchAny.zipfiles,
               geosearchAny.datas,
               gooey_languages,
               gooey_images,
               strip=None,
               upx=True,
               name='geosearch')