#!/usr/bin/env python
'''CanberraUAV utility functions for dealing with image regions'''

import numpy, sys, os, time, cuav_util, cv

class Region:
	'''a object representing a recognised region in an image'''
	def __init__(self, x1, y1, x2, y2, scan_shape, scan_score=0, compactness=0):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.latlon = None
		self.score = None
                self.scan_score = scan_score
                self.compactness = compactness
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
                cv.Rectangle(img, (max(x1-offset,0),max(y1-offset,0)), (x2+offset,y2+offset), colour, linewidth) 

        def __str__(self):
		return '%s latlon=%s score=%s' % (str(self.tuple()), str(self.latlon), self.score)
	    
def RegionsConvert(rlist, scan_shape, full_shape, calculate_compactness=True):
	'''convert a region list from tuple to Region format,
	also mapping to the shape of the full image'''
	ret = []
        scan_w = scan_shape[0]
        scan_h = scan_shape[1]
        full_w = full_shape[0]
        full_h = full_shape[1]
	for r in rlist:
		(x1,y1,x2,y2,score,pixscore) = r
		x1 = (x1 * full_w) // scan_w
		x2 = (x2 * full_w) // scan_w
		y1 = (y1 * full_h) // scan_h
		y2 = (y2 * full_h) // scan_h
                if calculate_compactness:
                        compactness = array_compactness(pixscore)
                else:
                        compactness = 0
		ret.append(Region(x1,y1,x2,y2, scan_shape, score, compactness))
	return ret

def array_compactness(im):
        '''
        calculate the compactness of a 2D array. Each element of the 2D array
        should be proportional to the score of that pixel in the overall scoring scheme
        . '''
        from numpy import array,meshgrid,arange,shape,mean,zeros
        from numpy import outer,sum,max,linalg
        from numpy import sqrt
        from math import exp
        (h,w) = shape(im)
        (X,Y) = meshgrid(arange(w),arange(h))
        x = X.flatten()
        y = Y.flatten()
        wgts = im[y,x]
        sw = sum(wgts)
        if sw == 0:
                return 1
        wgts /= sw
        wpts = array([wgts*x, wgts*y])
        wmean = sum(wpts, 1)
        N = len(x)
        s = array([x,y])
        P = zeros((2,2))
        for i in range(0,N):
                P += wgts[i]*outer(s[:,i],s[:,i])
        P = P - outer(wmean,wmean);

        det = abs(linalg.det(P))
        if (det <= 0):
                return 0.0
        v = linalg.eigvalsh(P)
        v = abs(v)
        r = min(v)/max(v)
        return 100.0*sqrt(r/det)

def image_whiteness(hsv):
        ''' a measure of the whiteness of an HSV image 0 to 1'''
        (width,height) = cv.GetSize(hsv)
        score = 0
        count = 0
        for x in range(width):
                for y in range(height):
                        (h,s,v) = hsv[y,x]
                        if (s < 25 and v > 50):
                                count += 1
        return float(count)/float(width*height)

def raw_hsv_score(hsv):
	'''try to score a HSV image based on hsv'''
	(width,height) = cv.GetSize(hsv)
	score = 0
	blue_count = 0
	red_count = 0
	sum_v = 0
	from numpy import zeros
	scorix = zeros((height,width))
	for x in range(width):
		for y in range(height):
			pix_score = 0
			(h,s,v) = hsv[y,x]
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

	return (score, scorix, blue_count, red_count, avg_v)

def hsv_score(r, hsv, use_compactness=False, use_whiteness=False):
	'''try to score a HSV image based on how "interesting" it is for joe detection'''
  	(score, scorix, blue_count, red_count, avg_v) = raw_hsv_score(hsv)

        r.hsv_score = score

	if blue_count < 100 and red_count < 50 and avg_v < 150:
		if blue_count > 1 and red_count > 1:
			score *= 2
		if blue_count > 2 and red_count > 2:
			score *= 2
		if blue_count > 4 and red_count > 4:
			score *= 2

        scorix = (scorix>0).astype(float)

	if use_compactness:
        	score = score*r.compactness

        r.whiteness = image_whiteness(hsv)
	if use_whiteness:
		not_white = 1.0-r.whiteness
		score = score*not_white
        score *= (1+r.scan_score)
        r.score = r.scan_score

def score_region(img, r, filter_type='simple'):
	'''filter a list of regions using HSV values'''
	(x1, y1, x2, y2) = r.tuple()
	if True:
		(w,h) = cuav_util.image_shape(img)
		x = (x1+x2)/2
		y = (y1+y2)/2
		x1 = max(x-10,0)
		x2 = min(x+10,w)
		y1 = max(y-10,0)
		y2 = min(y+10,h)
	cv.SetImageROI(img, (x1, y1, x2-x1,y2-y1))
	hsv = cv.CreateImage((x2-x1,y2-y1), 8, 3)
	cv.CvtColor(img, hsv, cv.CV_RGB2HSV)
	cv.ResetImageROI(img)
        if filter_type == 'compactness':
                use_compactness = True
        else:
                use_compactness = False
        hsv_score(r, hsv, use_compactness)

def filter_regions(img, regions, min_score=4, frame_time=None, filter_type='simple'):
	'''filter a list of regions using HSV values'''
	ret = []
	img = cv.GetImage(cv.fromarray(img))
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
    
