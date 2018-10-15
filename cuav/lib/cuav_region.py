#!/usr/bin/env python
'''CanberraUAV utility functions for dealing with image regions'''

import numpy, sys, os, time, cv2, math
from numpy import shape
from cuav.lib import cuav_util

class Region:
    '''a object representing a recognised region in an image'''
    def __init__(self, x1, y1, x2, y2, scan_shape, scan_score=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.latlon = None
        self.score = None
        self.scan_score = scan_score
        self.whiteness = None
        self.blue_score = None
        self.scan_shape = scan_shape

    def tuple(self):
        '''return the boundary as a tuple'''
        return (self.x1, self.y1, self.x2, self.y2)

    def center(self):
        '''return the center of the region'''
        return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)

    def draw_rectangle(self, img, colour=(255,0,0), linewidth=2, offset=2):
        '''draw a rectange around the region in an image'''
        (x1,y1,x2,y2) = self.tuple()
        (wview,hview) = cuav_util.image_shape(img)
        (w,h) = self.scan_shape
        x1 = x1*wview//w
        x2 = x2*wview//w
        y1 = y1*hview//h
        y2 = y2*hview//h
        cv2.rectangle(img, (max(x1-offset,0),max(y1-offset,0)), (x2+offset,y2+offset), colour, linewidth)

    def __str__(self):
        return '%s latlon=%s score=%s' % (str(self.tuple()), str(self.latlon), self.score)

def RegionsConvert(rlist, scan_shape, full_shape):
    '''convert a region list from tuple to Region format,
    also mapping to the shape of the full image'''
    ret = []
    scan_w = scan_shape[0]
    scan_h = scan_shape[1]
    full_w = full_shape[0]
    full_h = full_shape[1]
    for r in rlist:
        (x1,y1,x2,y2,score) = r
        x1 = (x1 * full_w) // scan_w
        x2 = (x2 * full_w) // scan_w
        y1 = (y1 * full_h) // scan_h
        y2 = (y2 * full_h) // scan_h

        ret.append(Region(x1,y1,x2,y2, scan_shape, score))
    return ret

def image_whiteness(hsv):
        ''' a measure of the whiteness of an HSV image 0 to 1'''
        #(width,height) = cv.GetSize(hsv)
        (height,width,d) = shape(hsv)
        score = 0
        count = 0
        if height == 0 or width == 0:
            return 0
        for x in range(width):
            for y in range(height):
                (h,s,v) = hsv[y,x]
                if (s < 25 and v > 50):
                    count += 1
        return float(count)/float(width*height)

def raw_hsv_score(hsv):
    '''try to score a HSV image based on hsv'''
    (height,width,d) = shape(hsv)
    score = 0
    blue_count = 0
    red_count = 0
    sum_v = 0

    # keep range of hsv
    h_min = 255
    h_max = 0
    s_min = 255
    s_max = 0
    v_min = 255
    v_max = 0

    from numpy import zeros
    scorix = zeros((height,width))
    for x in range(width):
        for y in range(height):
            pix_score = 0
            (h,s,v) = hsv[y,x]
            if h > h_max:
                    h_max = h
            if s > s_max:
                    s_max = s
            if v > v_max:
                    v_max = v
            if h < h_min:
                    h_min = h
            if s < s_min:
                    s_min = s
            if v < v_min:
                    v_min = v
            sum_v += v
            if (h < 22 or (h > 171 and h < 191)) and s > 50 and v < 150:
                pix_score += 3
                blue_count += 1
                #print (x,y),h,s,v,'B'
            elif h > 108 and h < 140 and s > 140 and v > 128:
                pix_score += 1
                red_count += 1
                #print (x,y),h,s,v,'R'
            elif h > 82 and h < 94 and s > 125 and v > 100 and v < 230:
                pix_score += 1
                red_count += 1
                #print (x,y),h,s,v,'Y'
            elif v > 160 and s > 100:
                pix_score += 1
                #print h,s,v,'V'
            elif h>70 and s > 110 and v > 90:
                pix_score += 1
                #print h,s,v,'S'
            score += pix_score
            scorix[y,x] = pix_score
    avg_v = sum_v / (width*height)
    score = 500 * float(score) / (width*height)
    score = max(score, 1)

    return (score, scorix, blue_count, red_count, avg_v, s_max - s_min, v_max - v_min)

def log_scaling(value, scale):
    # apply a log scaling to a value
    if value <= math.e:
        return scale
    return math.log(value)*scale

def hsv_score(r, hsv, use_whiteness=False):
    '''try to score a HSV image based on how "interesting" it is for joe detection'''
    (col_score, scorix, blue_count, red_count, avg_v, s_range, v_range) = raw_hsv_score(hsv)

    r.hsv_score = col_score

    if blue_count < 100 and red_count < 50 and avg_v < 150:
        if blue_count > 1 and red_count > 1:
            col_score *= 2
        if blue_count > 2 and red_count > 2:
            col_score *= 2
        if blue_count > 4 and red_count > 4:
            col_score *= 2

        scorix = (scorix>0).astype(float)

        r.whiteness = image_whiteness(hsv)
        if use_whiteness:
            not_white = 1.0-r.whiteness
            col_score *= not_white

    r.col_score = col_score
    if col_score <= math.e:
        scaled_col_score = 1
    else:
        scaled_col_score = math.log(col_score)

    (height,width,d) = shape(hsv)
    num_pixels = height*width
    if red_count > 15 and blue_count > 15:
        col_score *= 8
        
    # combine all the scoring systems
    r.score = r.scan_score*(s_range/128.0)*log_scaling(col_score,0.3)
    r.score = max(r.score, 1)

    #print(r.score, red_count, blue_count, num_pixels)

def rgb_score(img):
    '''count red and blue pixels for scoring. This is specially targeted
    at the CanberraUAV OBC 2018 target'''
    (height,width,d) = shape(img)
    num_pixels = width * height
    red_count=0
    blue_count=0
    col_thresh = 1.4
    for x in range(width):
        for y in range(height):
            (r,g,b) = img[y][x]
            if r > g*col_thresh and r > b*col_thresh:
                red_count += 1
            elif b > r*col_thresh and b > g*col_thresh:
                blue_count += 1
    threshold_pct = 3
    pct_red = 100.0 * red_count / num_pixels
    pct_blue = 100.0 * blue_count / num_pixels
    return min(pct_red, pct_blue) * 5000
    
def score_region(img, r, filter_type='simple'):
    '''filter a list of regions using HSV values'''
    (x1, y1, x2, y2) = r.tuple()
    (w,h) = cuav_util.image_shape(img)
    x = (x1+x2)//2
    y = (y1+y2)//2
    x1 = max(x-10,0)
    x2 = min(x+10,w)
    y1 = max(y-10,0)
    y2 = min(y+10,h)
    rimg = img[y1:y2, x1:x2]
    r.score = rgb_score(rimg)
    #hsv = cv2.cvtColor(rimg, cv2.COLOR_RGB2HSV)
    #hsv_score(r, hsv)
    #r.score += template_score(rimg)

def filter_regions(img, regions, min_score=4, filter_type='simple'):
    '''filter a list of regions using HSV values'''
    ret = []
    #img = cv.GetImage(cv.fromarray(img))
    for r in regions:
        if r.score is None:
            score_region(img, r, filter_type=filter_type)
        if r.score >= min_score:
            ret.append(r)
    return ret


def filter_boundary(regions, boundary, pos=None):
    '''filter a list of regions using a search boundary'''
    ret = []
    for r in regions:
        if pos is None:
            continue
        if pos.altitude < 10:
              r.score = 0
        #print pos
        if r.latlon is None or cuav_util.polygon_outside(r.latlon, boundary):
            r.score = 0
        ret.append(r)
    return ret

def filter_radius(regions, latlon, radius):
    '''filter a list of regions using a search boundary'''
    ret = []
    for r in regions:
        if r.latlon is None or cuav_util.gps_distance(latlon[0], latlon[1], r.latlon[0], r.latlon[1]) > radius:
          r.score = 0
        ret.append(r)
    return ret

def CompositeThumbnail(img, regions, thumb_size=100):
    '''extract a composite thumbnail for the regions of an image

    The composite will consist of N thumbnails side by side
    '''
    composite = []
    for i in range(len(regions)):
        (x1,y1,x2,y2) = regions[i].tuple()
        midx = (x1+x2)//2
        midy = (y1+y2)//2

        if (x2-x1) > thumb_size or (y2-y1) > thumb_size:
            # we need to shrink the region
            rsize = max(x2+1-x1, y2+1-y1)
            src = cuav_util.SubImage(img, (midx-rsize//2,midy-rsize//2,rsize,rsize))
            thumb = cv2.resize(src, (thumb_size, thumb_size))
        else:
            x1 = midx - thumb_size//2
            y1 = midy - thumb_size//2
            thumb = cuav_util.SubImage(img, (x1, y1, thumb_size, thumb_size))
        if composite == []:
            composite = thumb
        else:
            composite = numpy.concatenate((composite, thumb), axis=1)
    return composite
