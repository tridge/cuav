#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob

from cuav.image import scanner
from cuav.lib import cuav_util, cuav_region
from MAVProxy.modules.lib import mp_image, wxsettings, mp_settings
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
from MAVProxy.modules.lib.mp_menu import *

from optparse import OptionParser
parser = OptionParser("algtest.py [options] <file>")
(opts, args) = parser.parse_args()

slipmap = None

def show_image(view, selected_image, im_rgb):
  '''show a image view'''
  view.set_title(selected_image)
  if selected_image == "original":
    view.set_image(im_rgb)
    return
  im = cuav_util.LoadImage(selected_image + ".pnm")
  im_numpy = numpy.ascontiguousarray(cv.GetMat(im))
  im_rgb = cv.fromarray(im_numpy)
  view.set_image(im_rgb)

def process(args):
	'''process a file'''
        file_index = 0
        fname = args[file_index]

        settings = MPSettings(
          [
          MPSetting('MetersPerPixel', float, 0.2, range=(0,10), increment=0.01, digits=2, tab='Image Processing'),
          MPSetting('MinRegionArea', float, 0.15, range=(0,100), increment=0.05, digits=2),
          MPSetting('MaxRegionArea', float, 2.0, range=(0,100), increment=0.1, digits=1),
          MPSetting('MinRegionSize', float, 0.1, range=(0,100), increment=0.05, digits=2),
          MPSetting('MaxRegionSize', float, 2, range=(0,100), increment=0.1, digits=1),
          MPSetting('MaxRarityPct',  float, 0.02, range=(0,100), increment=0.01, digits=2),
          MPSetting('RegionMergeSize', float, 3.0, range=(0,100), increment=0.1, digits=1),
          MPSetting('minscore', int, 0, 'Min Score', range=(0,1000), increment=1, tab='Scoring'),
          MPSetting('filter_type', str, 'simple', 'Filter Type', choice=['simple', 'compactness']),
          MPSetting('brightness', float, 1.0, 'Display Brightness', range=(0.1, 10), increment=0.1,
                    digits=2, tab='Display')
          ],
          title='Settings'
          )

        menu = MPMenuSubMenu('View',
                             items=[
          MPMenuItem('Original Image', 'Original Image', 'originalImage'),
          MPMenuItem('Unquantised Image', 'Unquantised Image', 'unquantisedImage'),
          MPMenuItem('Thresholded Image', 'Thresholded Image', 'thresholdedImage'),
          MPMenuItem('Neighbours Image', 'Neighbours Image', 'neighboursImage'),
          MPMenuItem('Regions Image', 'Regions Image', 'regionsImage'),
          MPMenuItem('Pruned Image', 'Pruned Image', 'prunedImage'),
          MPMenuItem('Fit Window', 'Fit Window', 'fitWindow'),
          MPMenuItem('Full Zoom',  'Full Zoom', 'fullSize'),
          MPMenuItem('Next Image', 'Next Image', 'nextImage'),
          MPMenuItem('Previous Image',  'Previous Image', 'previousImage')])
        
        im_orig = cuav_util.LoadImage(fname)
        im_numpy = numpy.ascontiguousarray(cv.GetMat(im_orig))
        im_rgb = cv.fromarray(im_numpy)
        cv.CvtColor(im_rgb, im_rgb, cv.CV_BGR2RGB)

        # create the various views
        view = mp_image.MPImage(title='FullImage', can_zoom=True, can_drag=True)
        view.set_popup_menu(menu)

        dlg = wxsettings.WXSettings(settings)

        selected_image = "original"
        
        while dlg.is_alive() and view.is_alive():
        	last_change = settings.last_change()
                scan_parms = {
                  'MinRegionArea' : settings.MinRegionArea,
                  'MaxRegionArea' : settings.MaxRegionArea,
                  'MinRegionSize' : settings.MinRegionSize,
                  'MaxRegionSize' : settings.MaxRegionSize,
                  'MaxRarityPct'  : settings.MaxRarityPct,
                  'RegionMergeSize' : settings.RegionMergeSize,
                  'SaveIntermediate' : 1.0,
                  'MetersPerPixel' : settings.MetersPerPixel
                  }

                t0 = time.time()
                regions = scanner.scan(im_numpy, scan_parms)
                regions = cuav_region.RegionsConvert(regions,
                                                     cuav_util.image_shape(im_orig),
                                                     cuav_util.image_shape(im_orig), False)    
                t1=time.time()
                print("Processing took %.2f seconds" % (t1-t0))
                show_image(view, selected_image, im_rgb)

                while last_change == settings.last_change() and dlg.is_alive():
                	new_index = file_index
                	for event in view.events():
                        	if isinstance(event, MPMenuItem):
                                        if event.returnkey == 'nextImage':
                                        	new_index = (file_index + 1) % len(args)
                                        elif event.returnkey == 'previousImage':
                                        	new_index = (file_index - 1) % len(args)
                                	elif event.returnkey.endswith("Image"):
                                        	selected_image = event.returnkey[:-5]
                                                show_image(view, selected_image, im_rgb)
                        if new_index != file_index:
                        	file_index = new_index
                                fname = args[file_index]
                                im_orig = cuav_util.LoadImage(fname)
                                im_numpy = numpy.ascontiguousarray(cv.GetMat(im_orig))
                                im_rgb = cv.fromarray(im_numpy)
                                cv.CvtColor(im_rgb, im_rgb, cv.CV_BGR2RGB)
                                break
                        time.sleep(0.1)

# main program
process(args)
