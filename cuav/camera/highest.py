#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("edges.py [options] <filename>")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def mouse_event(event, x, y, flags, pgm):
    '''called on mouse events'''
    if flags & cv.CV_EVENT_FLAG_LBUTTON:
        print("[%u, %u] : %u" % (x, y, pgm.img[y, x]))

def circle_highest(filename):
    '''circle the highest value pixel in an image'''
    pgm = util.PGM(filename)
    maxpoint = pgm.array.argmax()
    maxpos = (maxpoint%1280, maxpoint/1280)
    color_img = cv.CreateImage((1280,960), 16, 3)

    cv.CvtColor(pgm.img, color_img, cv.CV_GRAY2RGB)

    overlay = cv.CreateImage((1280,960), 16, 3)
    cv.SetZero(overlay)

    cv.Circle(overlay, maxpos, 10, cv.CV_RGB(65535, 0, 0))

    cv.AddWeighted(color_img, 1.0, overlay, 1.0, 0.5, color_img)

    cv.ShowImage('Highest', color_img)
    cv.SetMouseCallback('Highest', mouse_event, pgm)

    return color_img


cv.NamedWindow('Highest')

i = 0
while True:
    print(args[i])
    image = circle_highest(args[i])
    i = util.key_menu(i, len(args), image, 'highest.png')

for f in args:
    circle_highest(f)

cv.DestroyWindow('Highest')
