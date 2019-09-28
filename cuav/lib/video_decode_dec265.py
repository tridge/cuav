#!/usr/bin/env python
'''
decode video using dec265 child process
'''

import time
import subprocess
import threading
import numpy
import cv2

try:
    # py3
    from queue import Queue, Empty
except ImportError:
    # py2
    from Queue import Queue, Empty

class VideoReader(object):
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        cmd = 'dec265 --buffer-size=512 - -o -'.split()
        err = open("dec.err", "wb")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=1024)
        self.q = Queue()
        t = threading.Thread(target=self.decode_loop)
        t.daemon = True
        t.start()

    def convert_from_yuv(self, b):
        '''convert an image from yuv format using convert subprocess'''
        b = self.p.stdout.read(self.width*self.height*3/2)
        cmd = "convert -interlace plane -depth 8 -sampling-factor 4:2:0 -size 300x300 yuv:- jpg:-".split()
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        jpg, stderr = p.communicate(bytearray(b))
        return cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), -1)

    def decode_yuv(self):
        '''decode yuv422 from dec265 child'''
        b = self.p.stdout.read(self.width*self.height*3/2)
        img = self.convert_from_yuv(b)
        self.q.put(img)

    def decode_loop(self):
        '''decode ppp ffpmeg child in a loop'''
        while True:
            self.decode_yuv()
        
    def get_image(self, data):
        '''get next image or None'''
        if len(data) > 0:
            self.p.stdin.write(data)
            self.p.stdin.flush()
        if self.q.empty():
            return None
        return self.q.get()

if __name__ == '__main__':
    import argparse
    import sys
    from MAVProxy.modules.lib import mp_image

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", type=str)
    args = ap.parse_args()

    decoder = VideoReader()

    counter = 0

    viewer = mp_image.MPImage(title='Decoded', width=200, height=200, auto_size=True)

    f = open(args.infile, "rb")
    while True:
        time.sleep(0.1)
        blk = f.read(512)
        if blk is None or len(blk) == 0:
            break
        print("Block %u" % counter)
        counter += 1
        img = decoder.get_image(blk)
        while img is not None:
            viewer.set_image(img)
            img = decoder.get_image('')
