#!/usr/bin/env python
'''
video encoder using x265 child process
'''

import cv2
import time
import subprocess
import select
import fcntl
import os
import io

class VideoWriter(object):
    def __init__(self, crop=None):
        cmd = 'x265 --input-res 300x300 - --input-depth 8 --fps 1 -o - --ref 1 --bframes 0 --keyint 20 --intra-refresh --constrained-intra --bitrate 4 --vbv-bufsize 4 --vbv-maxrate 6'.split()
        cmderr = "x265_enc.err"
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(cmderr,"w"), bufsize=1024)
        self.crop = None
        if crop:
            self.set_cropstr(crop)
        flags = fcntl.fcntl(self.p.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(self.p.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def convert_to_yuv(self, img):
        '''convert an image to yuv format using convert subprocess'''
        # first get as a jpeg byte array
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        result, encimg = cv2.imencode('.jpg', img, encode_param)

        # now use external convert command to get as yuv
        cmd = "convert - -colorspace YCbCr -sampling-factor 4:2:0 -interlace plane -depth 8 yuv:-".split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        yuv, stderr = p.communicate(bytearray(encimg))
        return yuv

    def set_cropstr(self, cropstr):
        '''setup cropping'''
        c = cropstr.split(",")
        if len(c) == 4:
            self.set_crop((int(c[0]), int(c[1]), int(c[2]), int(c[3])))
        else:
            print("Bad VideoWriter crop: ", cropstr)
            self.crop = None

    def set_crop(self, crop):
        '''setup cropping'''
        (x,y,w,h) = crop
        if w > 0 and h > 0:
            self.crop = crop
        else:
            self.crop = None

    def reset(self):
        '''reset deltas'''
        self.image = None
            
    def crop_image(self, img):
        '''crop image as requested'''
        if self.crop is None:
            return img
        (x,y,w,h) = self.crop
        return img[y:y+h,x:x+w]

    def report(self):
        pass
    
    def add_image(self, img, timestamp_ms):
        '''add an image to video, using delta encoding'''
        img = self.crop_image(img)
        yuv = self.convert_to_yuv(img)
        self.p.stdin.write(yuv)
        try:
            ret = os.read(self.p.stdout.fileno(), 65*1024)
        except Exception:
            return None
        return ret
