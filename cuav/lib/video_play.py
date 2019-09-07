#!/usr/bin/env python
'''
playback a video encoded as delta images
'''

import imutils
import cv2
import numpy
import time
import struct
from MAVProxy.modules.lib import mp_image

class VideoReader(object):
    def __init__(self):
        self.img = None

    def get_image(self, fin):
        '''get next image or None'''
        header = fin.read(10)
        if len(header) < 10:
            return (None,0)
        (enclen,x,y,dt) = struct.unpack("<IHHH", header)
        encimg = fin.read(enclen)
        barray = numpy.asarray(bytearray(encimg), dtype="uint8")
        jimg = cv2.imdecode(barray, 1)
        if self.img is None:
            self.img = jimg
        else:
            (height,width,depth) = jimg.shape
            self.img[y:y+height,x:x+width] = jimg
        return (self.img,dt)

if __name__ == '__main__':
    import argparse
    import sys
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", type=str, default=None, help='also output to avi file')
    ap.add_argument("--scale", type=float, default=None, help='scale displayed images')
    ap.add_argument("--speed", type=float, default=1.0, help='playback speed')
    ap.add_argument("infile", type=str, nargs='?')
    args = ap.parse_args()

    viewer = mp_image.MPImage(title='Image', width=200, height=200, auto_size=True)

    fin = open(args.infile, 'rb')
    vid = VideoReader()
    avi = None

    frame_num = 1

    while True:
        (img,dt) = vid.get_image(fin)
        if img is None:
            break
        if args.scale is not None:
            img = cv2.resize(img, (0,0), fx=args.scale, fy=args.scale)
        if dt > 0:
            print("Frame %u dt %u" % (frame_num, dt))
            frame_num += 1
            viewer.set_image(img)
            time.sleep(dt*0.001/args.speed)
            if args.avi is not None:
                if avi is None:
                    (height,width,depth) = img.shape
                    avi = cv2.VideoWriter(args.avi,cv2.VideoWriter_fourcc(*'XVID'), 1.0, (width,height))
                if avi is not None:
                    avi.write(img)

    if avi is not None:
        avi.release()
