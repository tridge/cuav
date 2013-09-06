#!/usr/bin/python

import numpy, os, time, cv, sys

from cuav.lib import cuav_util

from optparse import OptionParser
parser = OptionParser("colour.py [options]")
(opts, args) = parser.parse_args()

def mouse_event_zoom(event, x, y, flags, img):
    '''called on mouse events'''
    if flags & cv.CV_EVENT_FLAG_LBUTTON:
        print("[%u, %u] : %s" % (x, y, img[y, x]))

def mouse_event(event, x, y, flags, imgs):
    '''called on mouse events'''
    (rgb,img) = imgs
    if flags & cv.CV_EVENT_FLAG_LBUTTON:
        print("[%u, %u] : %s %s" % (x, y, img[y, x], rgb[y,x]))
        (h,s,v) = img[y,x]
        score = 0
        if h < 22 and s > 50:
            score += 3
            print h,s,v,'B'
        if h > 171 and h < 191 and s > 50:
            score += 3
            print h,s,v,'B2'
        if h > 120 and h < 200 and v > 50 and s > 100:
            score += 1
            print h,s,v,'R'
        if v > 160 and s > 100:
            score += (v-160)/10
            print h,s,v,'V'
        if h>70 and s > 110 and v > 50:
            score += 2
            print h,s,v,'S'
    if flags & cv.CV_EVENT_FLAG_RBUTTON:
        cv.SetImageROI(img, (x-16,y-16,32,32))
        scale = 8
        zoom_hsv = cv.CreateImage((32*scale,32*scale), 8, 3)
        zoom_rgb = cv.CreateImage((32*scale,32*scale), 8, 3)
        cv.Resize(img, zoom_hsv, cv.CV_INTER_NN)
        cv.ResetImageROI(img)
        (width,height) = (32*scale,32*scale)
        zoom_hsv2 = cv.CloneImage(zoom_hsv)
        for x in range(width):
            for y in range(height):
                (h,s,v) = zoom_hsv[y,x]
                s = 255
                zoom_hsv2[y,x] = (h,s,v)
        cv.CvtColor(zoom_hsv2, zoom_rgb, cv.CV_HSV2RGB)
        cv.ShowImage('Zoom', zoom_rgb)
        cv.SetMouseCallback('Zoom', mouse_event_zoom, zoom_hsv)

def spectrum():
    array = numpy.zeros((480,640,3),dtype='uint8')
    for y in range(480):
        for x in range(640):
            h = int((255*x)/640)
            s = 255
            v = int((255*y)/480)
            array[y,x] = (h,s,v)
    hsv = cv.GetImage(cv.fromarray(array))
    return hsv

if len(args) == 0:
    hsv = spectrum()
    rgb = cv.CreateImage(cv.GetSize(hsv), 8, 3)
    cv.CvtColor(hsv, rgb, cv.CV_HSV2RGB)
else:
    img = cuav_util.LoadImage(args[0])
    (w,h) = cv.GetSize(img)
    img2 = cv.CreateImage((w/2,h/2), 8, 3)
    cv.Resize(img, img2)
    hsv = cv.CreateImage((w/2,h/2), 8, 3)
    cv.CvtColor(img2, hsv, cv.CV_RGB2HSV)
    rgb = cv.CreateImage(cv.GetSize(hsv), 8, 3)
    cv.CvtColor(hsv, rgb, cv.CV_HSV2RGB)



cv.ShowImage('HSV', rgb)
cv.SetMouseCallback('HSV', mouse_event, (rgb,hsv))
cuav_util.cv_wait_quit()
