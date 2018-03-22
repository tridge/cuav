#!/usr/bin/env python

import sys, cv2, numpy, time
import os
import argparse

from cuav.lib import cuav_util
from cuav.image import scanner
from MAVProxy.modules.lib import mp_image
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
from MAVProxy.modules.lib.wxsettings import WXSettings
from MAVProxy.modules.lib.mp_menu import *


def show_mask(raw, w, h):
    '''show mask and min/max'''
    mask = 0
    minv = 65536
    maxv = 0
    raw = raw.byteswap(False)
    minv = numpy.min(raw)
    maxv = numpy.max(raw)
    print("Min=%u max=%u" % (minv, maxv))
    

def convert_image(filename, threshold, blue_threshold, green_threshold):
    '''convert a file'''
    im_orig = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    (w,h) = cuav_util.image_shape(im_orig)
    im2 = numpy.zeros((h,w,3),dtype='uint8')
    show_mask(im_orig, w, h)
    img = numpy.array(im_orig, dtype=numpy.uint16)
    img = numpy.ascontiguousarray(img)
    scanner.thermal_convert(img, im2, threshold, blue_threshold, green_threshold)
    return im2

def settings_callback(setting):
    '''called on a changed setting'''
    global changed
    changed = True


def show_value(x,y, filename):
    raw_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    try:
        v = raw_image[y][x]
    except Exception:
        return
    v1 = v>>8
    v2 = v&0xFF
    v3 = v2<<8 | v1
    print(x,y, v3)

def file_list(directory, extensions):
    '''return file list for a directory'''
    flist = []
    for (root, dirs, files) in os.walk(directory):
        for f in files:
            extension = f.split('.')[-1]
            if extension.lower() in extensions:
                flist.append(os.path.join(root, f))
    return sorted(flist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Thermal image viewer")
    parser.add_argument("imagedir", default=None, help='image directory')
    parser.add_argument("--threshold", type=int, default=6100, help="color threshold")
    args = parser.parse_args()
    
    if os.path.isdir(args.imagedir):
        files = file_list(args.imagedir, ['jpg', 'jpeg', 'png'])

    view_image = mp_image.MPImage(title='ThermalView',
                                  width=500,
                                  height=500,
                                  mouse_events=True,
                                  key_events=True,
                                  can_zoom=True,
                                  can_drag=True)

    menu = MPMenuTop([])
    view_menu = MPMenuSubMenu('View',
                              [MPMenuItem('Next Image\tCtrl+N', 'Next Image', 'nextImage'),
                               MPMenuItem('Previous Image\tCtrl+P', 'Previous Image', 'previousImage')
                              ])
    menu.add(view_menu)
    view_image.set_menu(menu)

    settings = MPSettings(
        [ MPSetting('threshold', int, args.threshold, 'High Threshold', tab='Settings', range=(0,65535)),
          MPSetting('blue_threshold', float, 0.75, 'Blue Threshold', range=(0,1)),
          MPSetting('green_threshold', float, 0.4, 'Green Threshold', range=(0,1))])

    changed = True
    
    settings.set_callback(settings_callback)

    WXSettings(settings)

    image_idx = 0

    while True:
        if changed:
            if image_idx >= len(files):
                image_idx = 0
            if image_idx < 0:
                image_idx = len(files)-1
            filename = files[image_idx]
            view_image.set_title('View: %s' % filename)
            color_img = convert_image(filename, settings.threshold, settings.blue_threshold, settings.green_threshold)
            view_image.set_image(color_img, bgr=True)
            changed = False
        if view_image.is_alive():
            for event in view_image.events():
                if isinstance(event, MPMenuGeneric):
                    if event.returnkey == 'nextImage':
                        image_idx += 1
                    elif event.returnkey == 'previousImage':
                        image_idx -= 1
                    changed = True
                elif event.ClassName == 'wxMouseEvent':
                    (x,y) = (event.X, event.Y)
                    show_value(x,y, filename)
        time.sleep(0.02)
