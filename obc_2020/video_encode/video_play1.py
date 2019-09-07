#!/usr/bin/env python
'''
playback a video encoded as delta images
'''

import argparse
import imutils
import cv2
import numpy
import sys
import time
import struct

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--delay", type=int, default=0)
ap.add_argument("--avi", type=str, default=None, help='also output to avi file')
ap.add_argument("--scale", type=float, default=None, help='scale displayed images')
ap.add_argument("infile", type=str, nargs='?')
args = ap.parse_args()


class VideoReader(object):
    def __init__(self, name):
        self.f = open(name, 'rb')
        self.img = None

    def get_image(self):
        '''get next image or None'''
        header = self.f.read(10)
        if len(header) < 10:
            return (None,0)
        (enclen,x,y,dt) = struct.unpack("<IHHH", header)
        encimg = self.f.read(enclen)
        barray = numpy.asarray(bytearray(encimg), dtype="uint8")
        jimg = cv2.imdecode(barray, 1)
        if self.img is None:
            self.img = jimg
        else:
            (height,width,depth) = jimg.shape
            self.img[y:y+height,x:x+width] = jimg
        return (self.img,dt)

    def close(self):
        '''close video file'''
        self.f.close()
        self.f = None

vid = VideoReader(args.infile)
avi = None

while True:
    (img,dt) = vid.get_image()
    if img is None:
        break
    if args.scale is not None:
        img = cv2.resize(img, (0,0), fx=args.scale, fy=args.scale)
    if dt > 0:
        cv2.imshow("Image", img)
        if args.delay > 0:
            cv2.waitKey(args.delay)
        else:
            cv2.waitKey(dt)
        if args.avi is not None:
            if avi is None:
                (height,width,depth) = img.shape
                avi = cv2.VideoWriter(args.avi,cv2.VideoWriter_fourcc(*'PIM1'), 1.0, (width,height))
            if avi is not None:
                avi.write(img)

if avi is not None:
    avi.release()
