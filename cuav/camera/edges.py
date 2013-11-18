#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("edges.py [options] <filename>")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def show_edges(filename):
    '''show ediges in an image'''
    pgm = util.PGM(filename)

    # convert to 8 bit
    img8 = cv.CreateImage((1280,960), 8, 1)
    cv.ConvertScale(pgm.img, img8, scale=1.0/256)

    edge1 = cv.CreateImage((1280,960), 8, 1)
    cv.Canny(img8, edge1, 250, 255, 5)

    edgecolor = cv.CreateImage((1280,960), 8, 3)
    edgecolor16 = cv.CreateImage((1280,960), 16, 3)
    cv.CvtColor(edge1, edgecolor, cv.CV_GRAY2RGB)
    cv.ConvertScale(edgecolor, edgecolor16, scale=256)

    color_img = cv.CreateImage((1280,960), 16, 3)
    cv.CvtColor(pgm.img, color_img, cv.CV_GRAY2RGB)

    cv.AddWeighted(color_img, 1.0, edgecolor16, 1.0, 0.5, color_img)

    cv.ShowImage('Edges', color_img)
    return color_img


cv.NamedWindow('Edges')

i = 0
while True:
    print(args[i])
    image = show_edges(args[i])
    i = util.key_menu(i, len(args), image, 'edges.png')

cv.DestroyWindow('Edges')
