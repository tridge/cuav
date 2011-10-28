#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("markimages.py [options] <filename>")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def mouse_event(event, x, y, flags, data):
    '''called on mouse events'''
    global idx, pgm
    if flags & cv.CV_EVENT_FLAG_LBUTTON:
        print("[%u, %u] : %u" % (x, y, pgm.img[y, x]))
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
        f = open('joe.txt', mode='a')
        f.write('%s %u %u\n' % (args[idx], x, y))
        f.close()
        print("Joe at %u,%u of %s" % (x, y, args[idx]))

def change_image(i):
    '''show image idx'''
    global idx, pgm
    idx = i
    pgm = util.PGM(args[idx])
    cv.ShowImage('CanberraUAV', pgm.img)

cv.NamedWindow('CanberraUAV')
if len(args) > 1:
    tbar = cv.CreateTrackbar('Image', 'CanberraUAV', 0, len(args)-1, change_image)
cv.SetMouseCallback('CanberraUAV', mouse_event, None)

idx = 0
pgm = None
while True:
    print(args[idx])
    change_image(idx)
    oldidx = idx
    newidx = util.key_menu(oldidx, len(args), None, None)
    idx += (newidx - oldidx)
    cv.SetTrackbarPos('Image', 'CanberraUAV', idx)

cv.DestroyWindow('CanberraUAV')
