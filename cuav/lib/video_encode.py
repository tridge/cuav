#!/usr/bin/env python
'''
video encoder that looks for structural differences and only sends changed image
areas that are above a threshold
'''

import imutils
import cv2
import struct
from skimage.measure import compare_ssim


class VideoWriter(object):
    def __init__(self, quality=50, min_area=8, threshold=225, crop=None):
        self.quality = quality
        self.min_area = min_area
        self.threshold = threshold
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
            
    def add_delta(self, img, x, y, dt):
        '''add a delta image located at x,y'''

        # encode delta
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), self.quality]
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
            return self.add_delta(self.image, 0, 0, 0)

        dt = timestamp_ms - self.timestamp_base_ms
        self.timestamp_base_ms = timestamp_ms

        gray1 = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")

        while True:
            thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh_inv = cv2.bitwise_not(thresh)

            # find contours
            cnts = cv2.findContours(thresh_inv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            min_area = max(min(self.min_area, self.largest_area(cnts)),4)
            num_areas = self.count_areas(cnts, min_area)
            count = 0
            if len(cnts) > 0 and len(cnts) < 5:
                break
            if len(cnts) == 0:
                if self.threshold > 230:
                    break
                self.threshold += 1
            if len(cnts) >= 5:
                if self.threshold < 50:
                    break
                self.threshold -= 1

        ret = bytearray()
        
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

            ret += self.add_delta(changed, x1, y1, dt_delta)

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
    ap.add_argument("--minarea", type=int, default=32)
    ap.add_argument("--threshold", type=int, default=225)
    ap.add_argument("--crop", type=str, default=None)
    ap.add_argument("imgs", type=str, nargs='+')
    args = ap.parse_args()

    if len(args.imgs) < 2:
        print("Need at least 2 images")
        sys.exit(1)
    
    # instantiate video delta encoder
    outf = open(args.outfile, 'wb')
    vid = VideoWriter(quality=args.quality, min_area=args.minarea, threshold=args.threshold, crop=args.crop)

    timestamp_ms = 0

    for f in args.imgs:
        img = cv2.imread(f)
        enc = vid.add_image(img, timestamp_ms)
        outf.write(enc)
        vid.report()
        timestamp_ms += 1000

    outf.close()
