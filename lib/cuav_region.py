#!/usr/bin/env python
'''CanberraUAV utility functions for dealing with image regions'''

import numpy, sys, os, time, cuav_util, cv

class Region:
	'''a object representing a recognised region in an image'''
	def __init__(self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.latlon = None
		self.score = None

        def tuple(self):
            '''return the boundary as a tuple'''
            return (self.x1, self.y1, self.x2, self.y2)

        def center(self):
            '''return the boundary as a tuple'''
            return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)

        def __str__(self):
		return '%s latlon=%s score=%u' % (str(self.tuple()), str(self.latlon), self.score)
	    
def RegionsConvert(rlist, width=640, height=480):
	'''convert a region list from tuple to Region format,
	also mapping to standard 1280x960'''
	ret = []
	for r in rlist:
		(x1,y1,x2,y2) = r
		x1 = (x1 * 1280) / width
		x2 = (x2 * 1280) / width
		y1 = (y1 * 960)  / height
		y2 = (y2 * 960)  / height
		ret.append(Region(x1,y1,x2,y2))
	return ret

def compactness(im):
  from numpy import array,meshgrid,arange,shape,mean,zeros
  from numpy import outer,sum,max,linalg
  from math import sqrt
  (h,w) = shape(im)
  (X,Y) = meshgrid(arange(w),arange(h))
  x = X.flatten()
  y = Y.flatten()
  wgts = im[y,x]
  wgts /= sum(wgts)
  wpts = array([wgts*x, wgts*y])
  wmean = sum(wpts, 1)
  N = len(x)
  s = array([x,y])
  P = zeros((2,2))
  for i in range(0,N):
    P += wgts[i]*outer(s[:,i],s[:,i])
  P = P - outer(wmean,wmean);

  det = abs(linalg.det(P))

  return 1.0/(1.0 + sqrt(det))

def hsv_score(hsv):
	'''try to score a HSV image based on how "interesting" it is for joe detection'''
	(width,height) = cv.GetSize(hsv)
	score = 0
	blue_count = 0
	red_count = 0
	sum_v = 0
	#from numpy import zeros
	#scorix = zeros((height,width))
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
			#scorix[y,x] = pix_score
	avg_v = sum_v / (width*height)
	#print blue_count, red_count, avg_v
	# apply compactness
	# NOTE: disabled until we find the memory leak on the panda
	#nessy=compactness(scorix)
	#score*=nessy
	if blue_count < 100 and red_count < 50 and avg_v < 150:
		if blue_count > 1 and red_count > 1:
			score *= 2
		if blue_count > 2 and red_count > 2:
			score *= 2
		if blue_count > 4 and red_count > 4:
			score *= 2
	return score

def score_region(img, r, min_score=4):
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
	r.score = hsv_score(hsv)

def filter_regions(img, regions, min_score=4, frame_time=None):
	'''filter a list of regions using HSV values'''
	ret = []
	img = cv.GetImage(cv.fromarray(img))
	for r in regions:
		if r.score is None:
			score_region(img, r)
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
    
