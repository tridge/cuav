#!/usr/bin/python

import numpy, os, time, cv2, sys, math, sys, glob
import argparse

from cuav.image import scanner
from cuav.lib import cuav_util, cuav_region
from MAVProxy.modules.lib import mp_image, wxsettings, mp_settings
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
from MAVProxy.modules.lib.mp_menu import *

slipmap = None

def show_image(view, selected_image, im_bgr, fname):
    '''show a image view'''
    view.set_title("%s %s" % (selected_image, fname))
    im = cv2.imread(selected_image,1)
    view.set_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

def file_list(directory, extensions):
    '''return file list for a directory'''
    flist = []
    for (root, dirs, files) in os.walk(directory):
        for f in files:
            extension = f.split('.')[-1]
            if extension.lower() in extensions:
                flist.append(os.path.join(root, f))
    return flist

def process(folderfile):
    '''process a file'''
    file_index = 0

    files = []
    if os.path.isdir(folderfile):
        files = file_list(folderfile, ['jpg', 'jpeg', 'png'])
    else:
        files.append(folderfile)

    fname = files[file_index]

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
        MPSetting('filter_type', str, 'simple', 'Filter Type', choice=['simple']),
        MPSetting('brightness', float, 1.0, 'Display Brightness', range=(0.1, 10), increment=0.1,
                digits=2, tab='Display')
    ],
    title='Settings'
    )

    menu = MPMenuSubMenu('View', items=[
        MPMenuItem('Original Image', 'Original Image', '_1original.pnm'),
        MPMenuItem('Unquantised Image', 'Unquantised Image', '_1unquantised.pnm'),
        MPMenuItem('Thresholded Image', 'Thresholded Image', '_2thresholded.pnm'),
        MPMenuItem('Neighbours Image', 'Neighbours Image', '_3neighbours.pnm'),
        MPMenuItem('Regions Image', 'Regions Image', '_4regions.pnm'),
        MPMenuItem('Prune Large Image', 'Prune Large Image', '_5prunelarge.pnm'),
        MPMenuItem('Merged Image', 'Merged Large', '_6merged.pnm'),
        MPMenuItem('Pruned Image', 'Pruned Image', '_7pruned.pnm'),
        MPMenuItem('Fit Window', 'Fit Window', 'fitWindow'),
        MPMenuItem('Full Zoom',  'Full Zoom', 'fullSize'),
        MPMenuItem('Next Image', 'Next Image', 'nextImage'),
        MPMenuItem('Previous Image',  'Previous Image', 'previousImage')
    ])
    
    im_orig = cv2.imread(fname,-1)
    im_numpy = numpy.ascontiguousarray(im_orig)

    # create the various views
    view = mp_image.MPImage(title='FullImage', can_zoom=True, can_drag=True)
    view.set_popup_menu(menu)

    dlg = wxsettings.WXSettings(settings)

    selected_image = "_1original.pnm"
    
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
        print("Processing %s took %.2f seconds" % (fname, t1-t0))
        show_image(view, str(file_index+1) + selected_image, im_orig, fname)
        

        while last_change == settings.last_change() and dlg.is_alive():
            new_index = file_index
            for event in view.events():
                if isinstance(event, MPMenuItem):
                    if event.returnkey == 'nextImage':
                        new_index = (file_index + 1) % len(files)
                    elif event.returnkey == 'previousImage':
                        new_index = (file_index - 1) % len(files)
                    elif event.returnkey.endswith("pnm"):
                        selected_image = event.returnkey
                        show_image(view, str(file_index+1) + selected_image, im_orig, fname)
                if new_index != file_index:
                    file_index = new_index
                    fname = files[file_index]
                    im_orig = cv2.imread(fname,-1)
                    im_numpy = numpy.ascontiguousarray(im_orig)
                    break
                time.sleep(0.1)
        #remove all the pnm's
        directory = os.getcwd()
        lst = os.listdir( directory )

        for item in lst:
            if item.endswith(".pnm"):
                os.remove(os.path.join(directory, item ))
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Algorithm tester")
    parser.add_argument("folder", default=None, help="Image folder or single file")
    args = parser.parse_args()
    
    # main program
    process(args.folder)
