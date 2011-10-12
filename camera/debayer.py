#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("debayer.py [options] <filename>")
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


cv.NamedWindow('Bayer')

i = 0
while True:
    print(args[i])
    image = debayer(args[i])
    i = util.key_menu(i, len(args), image, 'edges.png')

cv.DestroyWindow('Bayer')
