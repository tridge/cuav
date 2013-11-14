#!/usr/bin/env python

import sys, cv, numpy, time
import os

from cuav.image import scanner
from cuav.lib import cuav_util

from optparse import OptionParser
parser = OptionParser("debayer.py [options] <filename>")
parser.add_option("--batch", action='store_true', help="batch convert to png")
parser.add_option("--half", action='store_true', help="show half sized")
parser.add_option("--brightness", type='float', default=1.0, help="set brightness")
parser.add_option("--gamma", type='int', default=0, help="set gamma (if 16 bit images)")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def debayer(filename, show=True):
    '''debayer an image'''
    pgm = cuav_util.PGM(filename)
    img = numpy.zeros((960,1280,3),dtype='uint8')
    if opts.gamma != 0:
        img8 = numpy.zeros((960,1280,1),dtype='uint8')
        scanner.gamma_correct(pgm.array, img8, opts.gamma)
    else:
        img8 = pgm.array
    scanner.debayer(img8, img)
    color_img = cv.CreateImageHeader((1280, 960), 8, 3)
    cv.SetData(color_img, img)
    if opts.half:
        half_img = cv.CreateImage((640,480), 8, 3)
        cv.Resize(color_img, half_img)
        color_img = half_img        

    cv.ConvertScale(color_img, color_img, scale=opts.brightness)
    if show:
        cv.ShowImage('Bayer', color_img)
    return (color_img, pgm)

def mouse_event(event, x, y, flags, data):
    '''called on mouse events'''
    global idx, image
    if flags & cv.CV_EVENT_FLAG_LBUTTON:
        print("[%u, %u] : %s" % (x, y, image[y, x]))
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
        f = open('joe.txt', mode='a')
        f.write('%s %u %u\n' % (args[idx], x, y))
        f.close()
        print("Joe at %u,%u of %s" % (x, y, args[idx]))


def change_image(i):
    '''show image idx'''
    global idx
    idx = i
    return debayer(args[idx])

def show_images(args):
    '''show all images'''
    global image, idx

    cv.NamedWindow('Bayer')
    if len(args) > 1:
        tbar = cv.CreateTrackbar('Image', 'Bayer', 0, len(args)-1, change_image)
    cv.SetMouseCallback('Bayer', mouse_event, None)

    idx = 0
    pgm = None
    while True:
        print(args[idx])
        (image, pgm) = change_image(idx)
        oldidx = idx
        newidx = cuav_util.key_menu(oldidx, len(args), image,
                               '%s.png' % args[idx][:-4],
                               pgm=pgm)
        idx += (newidx - oldidx)
        cv.SetTrackbarPos('Image', 'Bayer', idx)
    cv.DestroyWindow('Bayer')

def convert_images(args):
    '''convert all images'''
    for f in args:
        png = f[:-4] + '.png'
        print("Saving %s" % png)
        (img, pgm) = debayer(f, False)
        cv.SaveImage(png, img)

if opts.batch:
    convert_images(args)
else:
    show_images(args)
    
