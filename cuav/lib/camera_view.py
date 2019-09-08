#!/usr/bin/env python

import picamera
import video_encode
import io
import numpy
import cv2
from MAVProxy.modules.lib import mp_image

camera = picamera.PiCamera()
viewer = mp_image.MPImage(title='Image', width=200, height=200, auto_size=True)

def cap_image_CV():
    '''capture one image'''
    s = io.BytesIO()
    camera.capture(s, "jpeg")
    s.seek(0)
    data = numpy.fromstring(s.getvalue(), dtype=numpy.uint8)
    return cv2.imdecode(data, 1)

import argparse
import sys
    
ap = argparse.ArgumentParser()
ap.add_argument("--crop", type=str, default=None)
ap.add_argument("--flipH", action='store_true', default=False)
ap.add_argument("--flipV", action='store_true', default=False)
args = ap.parse_args()

crop = None
if args.crop:
    c = args.crop.split(",")
    if len(c) == 4:
        crop = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
        

while True:
    img = cap_image_CV()
    if args.flipV:
        img = cv2.flip(img, 0)[:,:]
    if args.flipH:
        img = cv2.flip(img, 1)[:,:]
    if crop is not None:
        (x,y,w,h) = crop
        img = img[y:y+h,x:x+w]
    viewer.set_image(img,bgr=True)
