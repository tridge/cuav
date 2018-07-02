#!/usr/bin/env python
'''
test program for cuav_region
'''

import sys, os, time, random, functools, cv2
import pytest
import numpy as np
from cuav.lib import cuav_region, cuav_util
from cuav.lib.cuav_util import SubImage
from cuav.lib import mav_position


def test_Region():
    newregion = cuav_region.Region(10, 10, 25, 23, (10, 10), scan_score=450, compactness=10)

    assert newregion.tuple() == (10, 10, 25, 23)
    assert str(newregion) == "(10, 10, 25, 23) latlon=None score=None"
    assert newregion.center() == (17, 16)

    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    newregion.draw_rectangle(img)

def test_RegionsConvert():
    regions = []
    regions.append((200, 100, 204, 103, 200, np.array([[10, 15, 15, 10], [11, 13, 13, 9], [10, 12, 15, 11]], np.float32)))
    regions.append((250, 150, 254, 153, 10, np.array([[10, 12, 11, 10], [11, 9, 10, 9], [11, 14, 12, 10]], np.float32)))
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    regionsout = cuav_region.RegionsConvert(regions, cuav_util.image_shape(im_orig), cuav_util.image_shape(im_orig), calculate_compactness=True)
    assert len(regionsout) == 2

#def test_array_compactness():
#    subimage = np.array([[10, 15, 15, 10], [11, 13, 13, 9], [10, 12, 15, 11]], np.float32)
#    cpt = cuav_region.array_compactness(subimage)
#    assert cpt > 0 and cpt < 1000

def test_image_whiteness():
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    region = (1020, 658, 30, 28)
    subimage = SubImage(im_orig, region)
    hsvsubimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2HSV)
    wht = cuav_region.image_whiteness(hsvsubimage)
    assert wht > 0
    assert wht < 1

def test_raw_hsv_score():
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    region = (1020, 658, 30, 28)
    subimage = SubImage(im_orig, region)
    hsvsubimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2HSV)
    score = cuav_region.raw_hsv_score(hsvsubimage)
    assert score[0] > 0 and score[0] < 500
    assert score[1].shape == (28, 30)
    assert score[2] > 0
    assert score[3] > 0
    assert score[4] > 0
    assert score[5] > 0
    assert score[6] > 0

def test_log_scaling():
    assert cuav_region.log_scaling(20, 2) == 5.991464547107982
    assert cuav_region.log_scaling(1, 20) == 20
    assert cuav_region.log_scaling(3, 5) == 5.493061443340549

def test_hsv_score():
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    region = (1020, 658, 30, 28)
    subimage = SubImage(im_orig, region)
    hsvsubimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2HSV)
    newregion = cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20)

    cuav_region.hsv_score(newregion, hsvsubimage, False)
    assert newregion.hsv_score > 0
    assert newregion.whiteness > 0
    assert newregion.compactness > 0
    assert newregion.score > 0

def test_score_region():
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    newregion = cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20)
    cuav_region.score_region(im_orig, newregion, filter_type='simple')
    assert newregion.hsv_score > 0
    assert newregion.whiteness > 0
    assert newregion.compactness > 0
    assert newregion.score > 0

def test_filter_regions():
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    regions = []
    regions.append(cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20))
    regions.append(cuav_region.Region(30, 54, 50, 74, (20, 20), compactness=2.1, scan_score=15))
    ret = cuav_region.filter_regions(im_orig, regions, filter_type='compactness')
    assert len(ret) == 1
    assert ret[0].scan_score == 20
    assert ret[0].score > 4
    assert ret[0].compactness == 5.4

def test_filter_boundary():
    OBC_boundary = cuav_util.polygon_load(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    regions = []
    regOne = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regOne.latlon=(-26.6398870, 151.8220000)
    regOne.score = 20
    regions.append(regOne)
    regTwo = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regTwo.score = 32
    regTwo.latlon=(-26.6418700, 151.8709260)
    regions.append(regTwo)
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    ret = cuav_region.filter_boundary(regions, OBC_boundary, pos)
    assert len(ret) == 2
    assert ret[0].score == 0
    assert ret[1].score == 32


def test_filter_radius():
    regions = []
    regOne = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regOne.latlon=(-26.6398870, 151.8220000)
    regOne.score = 20
    regions.append(regOne)
    regTwo = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regTwo.score = 32
    regTwo.latlon=(-26.6418700, 151.8709260)
    regions.append(regTwo)
    ret = cuav_region.filter_radius(regions, (-26.6415, 151.8715), 200)
    assert len(ret) == 2
    assert ret[0].score == 0
    assert ret[1].score == 32

def test_CompositeThumbnail():
    regions = []
    regions.append(cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20))
    regions.append(cuav_region.Region(30, 54, 50, 74, (20, 20), compactness=2.1, scan_score=15))
    regions.append(cuav_region.Region(60, 24, 170, 134, (110, 110), compactness=0, scan_score=0))
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    composite = cuav_region.CompositeThumbnail(im_orig, regions)
    assert cuav_util.image_shape(composite) == (300, 100)
