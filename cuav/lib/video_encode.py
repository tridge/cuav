#!/usr/bin/env python
'''
video encoder that looks for structural differences and only sends changed image
areas
'''

import imutils
import cv2
import struct
import numpy
from skimage.measure import compare_ssim
from MAVProxy.modules.lib import mp_image

class VideoWriter(object):
    def __init__(self, initial_quality=20, quality=50, min_area=8, crop=None):
        self.initial_quality = initial_quality
        self.quality = quality
        self.min_area = min_area
        self.total_size = 0
        self.num_frames = 0
        self.num_deltas = 0
        self.image = None
        self.last_image = None
        self.shape = None
        self.timestamp_base_ms = 0
        self.crop = None
        if crop:
            self.set_cropstr(crop)

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
            
    def add_delta(self, img, x, y, dt, quality=None):
        '''add a delta image located at x,y'''
        if quality is None:
            quality = self.quality

        # encode delta
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        result, encimg = cv2.imencode('.webp', img, encode_param)

        enclen = len(encimg)
        header = struct.pack("<IHHH", enclen, x, y, dt)
        self.total_size += len(header) + len(encimg)
        self.num_deltas += 1
        if dt > 0 or self.num_frames == 0:
            self.num_frames += 1
        return bytearray(header) + bytearray(encimg)

    def largest_area(self, cnts):
        '''find largest contour by area'''
        largest_area = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            if area > largest_area:
                largest_area = area
        return largest_area

    def count_areas(self, cnts, min_area):
        '''work out how many deltas we will send'''
        count = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            if area >= min_area:
                count += 1
        return count

    def crop_image(self, img):
        '''crop image as requested'''
        if self.crop is None:
            return img
        (x,y,w,h) = self.crop
        return img[y:y+h,x:x+w]
    
    def add_image(self, img, timestamp_ms):
        '''add an image to video, using delta encoding'''
        img = self.crop_image(img)
        if self.image is None:
            # initial image
            self.shape = img.shape
            self.image = img.copy()
            self.last_image = self.image
            self.timestamp_base_ms = timestamp_ms
            return self.add_delta(self.image, 0, 0, 0, quality=self.initial_quality)

        dt = timestamp_ms - self.timestamp_base_ms
        self.timestamp_base_ms = timestamp_ms

        threshold = None
        ret = bytearray()

        delta = None
        count = 0
        max_count = 4

        while True:
            gray1 = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            (score, diff) = compare_ssim(gray1, gray2, full=True)
            minvalue = numpy.amin(diff)
            if threshold is None:
                threshold = min(1.05*(minvalue + 0.02), 0.9)
            elif minvalue > threshold or count > max_count:
                break

            if delta is not None:
                ret += self.add_delta(delta[0], delta[1], delta[2], 0)
                delta = None

            minloc = numpy.where(diff == minvalue)
            y = int(minloc[0])
            x = int(minloc[1])

            w = 16
            x = max(x-w//2,0)
            y = max(y-w//2,0)

            # expand area
            (x1, y1, w, h) = (x, y, w, w)
            x2 = x1 + w
            y2 = y1 + h
            (height, width, depth) = img.shape
            x2 = min(x2, width)
            y2 = min(y2, height)

            # cut out the changed area from image
            changed = img[y1:y2,x1:x2]

            #print(minvalue, threshold, x, y)

            # overwrite the current image with that area
            self.image[y1:y2,x1:x2] = changed

            delta = (changed, x1, y1)
            count += 1
            self.last_image[y1:y2,x1:x2] = changed

        if delta is not None:
            ret += self.add_delta(delta[0], delta[1], delta[2], dt)
        self.last_image = img.copy()
        return ret
        
    def report(self):
        '''show encoding size'''
        print("Encoded %u frames %u deltas at %u bytes/frame" % (self.num_frames, self.num_deltas, self.total_size/self.num_frames))

if __name__ == '__main__':
    import argparse
    import sys
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", type=str, default='out.cvid')
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--quality", type=int, default=50)
    ap.add_argument("--initial-quality", type=int, default=20)
    ap.add_argument("--minarea", type=int, default=32)
    ap.add_argument("--crop", type=str, default=None)
    ap.add_argument("imgs", type=str, nargs='+')
    args = ap.parse_args()

    if len(args.imgs) < 2:
        print("Need at least 2 images")
        sys.exit(1)
    
    # instantiate video delta encoder
    outf = open(args.outfile, 'wb')
    vid = VideoWriter(initial_quality=args.initial_quality, quality=args.quality, min_area=args.minarea, crop=args.crop)

    timestamp_ms = 0

    for f in args.imgs:
        img = cv2.imread(f)
        enc = vid.add_image(img, timestamp_ms)
        outf.write(enc)
        vid.report()
        timestamp_ms += 1000

    outf.close()
