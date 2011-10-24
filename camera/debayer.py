#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("debayer.py [options] <filename>")
parser.add_option("--batch", dest="batch", action='store_true', help="batch convert to png")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def debayer(filename):
    '''debayer an image'''
    pgm = util.PGM(filename)

    img8 = cv.CreateImage((1280,960), 8, 1)
    cv.ConvertScale(pgm.img, img8, scale=1.0/256)
    
    color_img = cv.CreateImage((1280,960), 8, 3)
    cv.CvtColor(img8, color_img, cv.CV_BayerGR2BGR)

    cv.ShowImage('Bayer', color_img)
    return color_img

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
    global idx, image
    idx = i
    image = debayer(args[idx])
    cv.ShowImage('Bayer', image)
    return image

def show_images(args):
    '''show all images'''
    global image, idx

    cv.NamedWindow('Bayer')
    tbar = cv.CreateTrackbar('Image', 'Bayer', 0, len(args)-1, change_image)
    cv.SetMouseCallback('Bayer', mouse_event, None)

    idx = 0
    pgm = None
    while True:
        print(args[idx])
        image = change_image(idx)
        oldidx = idx
        newidx = util.key_menu(oldidx, len(args), image,
                               '%s.png' % args[idx][:-4])
        idx += (newidx - oldidx)
        cv.SetTrackbarPos('Image', 'Bayer', idx)
    cv.DestroyWindow('Bayer')

def convert_images(args):
    '''convert all images'''
    for f in args:
        png = f[:-4] + '.png'
        print("Saving %s" % png)
        img = debayer(f)
        cv.SaveImage(png, img)

if opts.batch:
    convert_images(args)
else:
    show_images(args)
    
