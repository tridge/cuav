#!/usr/bin/env python

import sys, cv, numpy, time
import util

from optparse import OptionParser
parser = OptionParser("edges.py [options] <filename>")
(opts, args) = parser.parse_args()

if len(args) < 1:
    print("please supply an image file name")
    sys.exit(1)

def change_threshold(value):
    '''change displayed threshold'''
    global img_thresh, imgf, threshold
    threshold = value
    cv.Threshold(imgf, img_thresh, value/65536.0, 0, cv.CV_THRESH_TOZERO)
    cv.ShowImage('Threshold', img_thresh)
    return img_thresh

def show_threshold(filename):
    '''threshold an image'''
    global threshold, imgf
    pgm = util.PGM(filename)

    cv.ConvertScale(pgm.img, imgf, scale=1.0/65536)
    return change_threshold(threshold)

imgf = cv.CreateImage((1280,960), cv.IPL_DEPTH_32F, 1)
img_thresh = cv.CreateImage((1280,960), cv.IPL_DEPTH_32F, 1)
threshold = 0

cv.NamedWindow('Threshold')
cv.CreateTrackbar('Threshold', 'Threshold', 0, 65536, change_threshold)

i = 0
while True:
    print(args[i])
    image = show_threshold(args[i])
    i = util.key_menu(i, len(args), image, 'threshold.png')

cv.DestroyWindow('Threshold')
