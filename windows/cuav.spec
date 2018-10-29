# -*- mode: python -*-
# spec file for pyinstaller to build cuav for windows
from PyInstaller.utils.hooks import collect_submodules 

import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix = 'gooey/images')

import MAVProxy.modules.mavproxy_map
map_root = os.path.dirname(MAVProxy.modules.mavproxy_map.__file__)
map_data = Tree(os.path.join(map_root, 'data'), prefix = 'MAVProxy/modules/mavproxy_map/data')

import cuav.image.scanner
cuav_root = os.path.dirname(cuav.image.scanner.__file__)
cuav_data = Tree(os.path.join(cuav_root, ''), prefix = 'cuav/image')

geotagAny = Analysis(['..\\cuav\\tools\\geotag.py'],
             pathex=[os.path.abspath('.')],
             # for some unknown reason these hidden imports don't pull in
             # all the needed pieces, so we also import them in mavproxy.py
             hiddenimports=['cv2', 'wx', 'pylab', 
                            'numpy', 'dateutil', 'matplotlib',
                            'wx.grid', 'wx._grid',
                            'wx.lib.agw.genericmessagedialog', 'wx.lib.wordwrap', 'wx.lib.buttons',
                            'wx.lib.embeddedimage', 'wx.lib.imageutils', 'wx.lib.agw.aquabutton', 
                            'wx.lib.agw.gradientbutton',
                            'six','packaging', 'packaging.version', 'packaging.specifiers'] + collect_submodules('pymavlink'),
             excludes=['tcl', 'tk', 'Tkinter', 'tkinter', '_tkinter', 'sphinx', 'docutils', 'alabaster'],
             hookspath=None,
             runtime_hooks=None)

geosearchAny = Analysis(['..\\cuav\\tools\\geosearch.py'],
             pathex=[os.path.abspath('.')],
             # for some unknown reason these hidden imports don't pull in
             # all the needed pieces, so we also import them in mavproxy.py
            hiddenimports=['cv2', 'wx', 'pylab', 
                            'numpy', 'dateutil', 'matplotlib',
                            'wx.grid', 'wx._grid',
                            'wx.lib.agw.genericmessagedialog', 'wx.lib.wordwrap', 'wx.lib.buttons',
                            'wx.lib.embeddedimage', 'wx.lib.imageutils', 'wx.lib.agw.aquabutton', 
                            'wx.lib.agw.gradientbutton',
                            'six','packaging', 'packaging.version', 'packaging.specifiers'] + collect_submodules('pymavlink') +  collect_submodules('MAVProxy'),
             excludes=['tcl', 'tk', 'Tkinter', 'tkinter', '_tkinter', 'sphinx', 'docutils', 'alabaster'],
             hookspath=None,
             runtime_hooks=None)
             
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
               map_data,
               cuav_data,
               strip=None,
               upx=True,
               name='geosearch')