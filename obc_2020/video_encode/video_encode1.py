#!/usr/bin/env python
'''
video encoder that looks for structural differences and only sends changed image
areas that are above a threshold
'''

import argparse
import imutils
import cv2
import sys
import struct
from skimage.measure import compare_ssim

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--outfile", type=str, default='out.cvid')
ap.add_argument("--delay", type=int, default=0)
ap.add_argument("--quality", type=int, default=50)
ap.add_argument("--minarea", type=int, default=32)
ap.add_argument("--threshold", type=int, default=100)
ap.add_argument("imgs", type=str, nargs='+')
args = ap.parse_args()

if len(args.imgs) < 2:
    print("Need at least 2 images")
    sys.exit(1)

class VideoWriter(object):
    def __init__(self, outname, quality=50, min_area=8, threshold=100):
        self.quality = quality
        self.min_area = min_area
        self.threshold = threshold
        self.f = open(outname, 'wb')
        self.total_size = 0
        self.num_frames = 0
        self.num_deltas = 0
        self.image = None
        self.last_image = None
        self.shape = None
        self.timestamp_base_ms = 0

    def add_delta(self, img, x, y, dt):
        '''add a delta image located at x,y'''

        # encode delta as jpeg
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)

        enclen = len(encimg)
        header = struct.pack("<IHHH", enclen, x, y, dt)
        self.f.write(header)
        self.f.write(encimg)
        self.total_size += len(header) + len(encimg)
        self.num_deltas += 1
        if dt > 0 or self.num_frames == 0:
            self.num_frames += 1

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
    
    def add_image(self, img, timestamp_ms):
        '''add an image to video, using delta encoding'''
        if self.image is None:
            # initial image
            self.shape = img.shape
            self.image = img.copy()
            self.last_image = self.image
            self.timestamp_base_ms = timestamp_ms
            self.add_delta(self.image, 0, 0, 0)
            return

        dt = timestamp_ms - self.timestamp_base_ms
        self.timestamp_base_ms = timestamp_ms

        gray1 = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh_inv = cv2.bitwise_not(thresh)

        # find contours
        cnts = cv2.findContours(thresh_inv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        min_area = max(min(self.min_area, self.largest_area(cnts)),4)
        num_areas = self.count_areas(cnts, min_area)
        count = 0

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            if area < min_area:
                continue
            # expand area
            expansion = 5
            (height,width,depth) = self.shape
            (x1, y1, w, h) = cv2.boundingRect(c)
            x2 = x1 + w
            y2 = y1 + h
            x1 = max(x1-expansion, 0)
            y1 = max(y1-expansion, 0)
            x2 = min(x2+expansion, width)
            y2 = min(y2+expansion, height)

            # cut out the changed area from image
            changed = img[y1:y2,x1:x2]

            # overwrite the current image with that area
            self.image[y1:y2,x1:x2] = changed

            count += 1
            if count < num_areas:
                dt_delta = 0
            else:
                dt_delta = dt

            vid.add_delta(changed, x1, y1, dt_delta)

        self.last_image = img.copy()
        
    def close(self):
        '''close video file'''
        self.f.close()
        self.f = None

    def report(self):
        '''show encoding size'''
        print("Encoded %u frames %u deltas at %u bytes/frame" % (self.num_frames, self.num_deltas, self.total_size/self.num_frames))

# instantiate video delta encoder
vid = VideoWriter(args.outfile, quality=args.quality, min_area=args.minarea, threshold=args.threshold)

# load first image
image1 = cv2.imread(args.imgs[0])
image = image1

# get image shape
(height,width,depth) = image.shape

timestamp_ms = 0

for f in args.imgs[1:]:
    img = cv2.imread(f)
    vid.add_image(img, timestamp_ms)
    vid.report()
    timestamp_ms += 1000

vid.close()
