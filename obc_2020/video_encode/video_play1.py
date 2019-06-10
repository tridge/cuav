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
ap.add_argument("--delay", type=int, default=1000)
ap.add_argument("vidfile", type=str, nargs='?')
args = ap.parse_args()


class VideoReader(object):
    def __init__(self, name):
        self.f = open(name, 'rb')
        self.img = None

    def get_image(self):
        '''get next image or None'''
        header = self.f.read(10)
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

vid = VideoReader(args.vidfile)

while True:
    (img,dt) = vid.get_image()
    if dt > 0:
        cv2.imshow("Image", img)
        cv2.waitKey(dt)
