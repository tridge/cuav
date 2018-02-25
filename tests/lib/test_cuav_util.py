#!/usr/bin/env python
'''
tests for cuav_util.py
'''

import sys, os, time, random, functools, math, cv2
import pytest
from cuav.lib.cuav_util import *
from cuav.camera.cam_params import CameraParams
from cuav.lib import mav_position, cuav_region
            

def test_gps_distance():
    dist = gps_distance(-50.5, 145.34, -50.51, 145.37)
    assert abs(2398 - dist) < 1

    dist = gps_distance(-50.5, 145.34, -50.1, 145.1)
    assert abs(47685 - dist) < 1
    
    dist = gps_distance(-50.5, 145.34, -50.5, 145.35)
    assert abs(708 - dist) < 1

def test_gps_bearing():
    bearing = gps_bearing(-50.5, 145.34, -50.51, 145.37)
    assert abs(117.7 - bearing) < 0.1

    bearing = gps_bearing(-50.5, 145.34, -50.1, 145.1)
    assert abs(338.9 - bearing) < 0.1
    
    bearing = gps_bearing(-50.5, 145.34, -50.5, 145.35)
    assert abs(90.0 - bearing) < 0.1
    
def test_gps_newpos():
    (newlat, newlon) = gps_newpos(-50.5, 145.34, 0, 1000)
    assert abs(-50.49102 - newlat) < 0.00001
    assert abs(145.34 - newlon) < 0.00001

    (newlat, newlon) = gps_newpos(-50.5, 145.34, 90, 2.5)
    assert abs(-50.5 - newlat) < 0.00001
    assert abs(145.34003 - newlon) < 0.00001
    
    (newlat, newlon) = gps_newpos(-50.5, 145.34, 300, 5500)
    assert abs(-50.47527 - newlat) < 0.001
    assert abs(145.27276 - newlon) < 0.001

def test_angle_of_view():
    angle = angle_of_view(lens=4.0, sensorwidth=5.0)
    assert abs(64.01076 - angle) < 0.00001
    
    angle = angle_of_view(lens=4.0, sensorwidth=2.5)
    assert abs(34.70804 - angle) < 0.00001
    
def test_groundwidth():
    width = groundwidth(100, 4.0, 5.0)
    assert abs(125 - width) < 0.01
    
    width = groundwidth(56.7, 4.0, 2.5)
    assert abs(35.44 - width) < 0.01
    
def test_pixel_width():
    width = pixel_width(100, 3000, 4.0, 5.0)
    assert abs(0.042 - width) < 0.001
    
    width = pixel_width(56.7, 1024, 4.0, 2.5)
    assert abs(0.035 - width) < 0.001

def test_pixel_height():
    height = pixel_height(100, 2000, 4.0, 5.0)
    assert abs(0.062 - height) < 0.001
    
    height = pixel_height(56.7, 768, 4.0, 2.5)
    assert abs(0.046 - height) < 0.001
    
def test_pixel_position_matt():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=768)
    (east, north) = pixel_position_matt(100, 100, 123, 0.1, 2, 0, C)
    assert abs(-67.3798 - east) < 0.001
    assert abs(43.6719 - north) < 0.001
    
    (east, north) = pixel_position_matt(0, 130, 57, 0.1, 2, 0, C)
    assert abs(-38.4761 - east) < 0.001
    assert abs(18.188 - north) < 0.001

def test_pixel_coordinates():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=800)
    (lat, lon) = pixel_coordinates(512, 400, -50, 145, 123, 0, 0, 90, C)
    assert abs(-50 - lat) < 0.00001
    assert abs(145 - lon) < 0.00001
    
    (lat, lon) = pixel_coordinates(200, 100, -50, 145, 123, 2, 5, 50, C)
    assert abs(-49.99928 - lat) < 0.00001
    assert abs(145.00001 - lon) < 0.00001
    
    
def test_gps_position_from_xy():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=800)
    frame_time = 1478954763.0
    pos = mav_position.MavPosition(-50, 145, 120, 0, 0, 90, frame_time)
    (lat, lon) = gps_position_from_xy(512, 400, pos, C=C, altitude=120, shape=None)
    assert abs(-50 - lat) < 0.00001
    assert abs(145 - lon) < 0.00001
    
    (lat, lon) = gps_position_from_xy(200, 100, pos, C=C, altitude=120, shape=None)
    assert abs(-49.999589 - lat) < 0.00001
    assert abs(145.000614 - lon) < 0.00001
    
    
def test_meters_per_pixel():
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=800)
    frame_time = 1478954763.0
    pos = mav_position.MavPosition(-50, 145, 120, 0, 0, 90, frame_time)
    ret = meters_per_pixel(pos, C)
    assert abs(0.1463 - ret) < 0.001
    
    pos = mav_position.MavPosition(-50, 145, 85, 3, 1, 45, frame_time)
    ret = meters_per_pixel(pos, C)
    assert abs(0.1041 - ret) < 0.001
    
def test_gps_position_from_image_region():
    frame_time = 1478954763.0
    pos = mav_position.MavPosition(-50, 145, 120, 0, 0, 90, frame_time)
    region = cuav_region.Region(10, 10, 25, 23, None, scan_score=450, compactness=10)
    C = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1024, yresolution=800)
    (lat, lon) = gps_position_from_image_region(region, pos, width=1024, height=800, C=C, altitude=None)
    assert abs(-49.99934 - lat) < 0.00001
    assert abs(145.00078 - lon) < 0.00001
    
def test_mkdir_p():
    dirry = os.path.join(os.getcwd(), 'tests1')
    assert os.path.isdir(dirry) == False
    os.mkdir(dirry)
    assert os.path.isdir(dirry) == True
    mkdir_p(dirry)
    os.rmdir(dirry)
    assert os.path.isdir(dirry) == False
    mkdir_p(dirry)
    assert os.path.isdir(dirry) == True
    os.rmdir(dirry)
    assert os.path.isdir(dirry) == False
    
def test_frame_time():
    frametime = 1478954763.0
    assert frame_time(frametime) == "2016111212460300Z"
    
def test_parse_frame_time():
    filename = "img2016111212460300Z.png"
    assert parse_frame_time(filename) == 1478954763.0

def test_datetime_to_float():
    frame_time = datetime.datetime.strptime('2016111223465695Z', "%Y%m%d%H%M%S%fZ")
    assert datetime_to_float(frame_time) == 1478994416.95
    frame_time = datetime.datetime.strptime('2018111223965695Z', "%Y%m%d%H%M%S%fZ")
    assert datetime_to_float(frame_time) == 1542064146.5695
    
def test_polygon_outside():
    OBC_boundary = polygon_load(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    test_points = [
        (-26.6398870, 151.8220000, True ),
        (-26.6418700, 151.8709260, False ),
        (-350000000, 1490000000, True ),
        (0, 0,                   True ),
        (-26.5768150, 151.8408250, False ),
        (-26.5774060, 151.8405860, True ),
        (-26.6435630, 151.8303440, True ),
        (-26.6435650, 151.8313540, False ),
        (-26.6435690, 151.8303530, False ),
        (-26.6435690, 151.8303490, True ),
        (-26.5875990, 151.8344049, True ),
        (-26.6454781, 151.8820530, True ),
        (-26.6454779, 151.8820530, True ),
        (-26.6092109, 151.8747420, True ),
        (-26.6092111, 151.8747420, False ),
        (-26.6092110, 151.8747421, True ),
        (-26.6092110, 151.8747419, False ),
        (-26.6092111, 151.8747421, True ),
        (-26.6092109, 151.8747421, True ),
        (-26.6092111, 151.8747419, False ),
        (-27.6092111, 151.8747419, True ),
        (-27.6092111, 152.0000000, True ),
        (-25.0000000, 150.0000000, True )
    ]
    for lat, lon, outside in test_points:
        assert outside == polygon_outside((lat, lon), OBC_boundary)

def test_polygon_complete():
    OBC_boundary = polygon_load(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    assert polygon_complete(OBC_boundary)
    
def test_polygon_load():
    OBC_boundary = polygon_load(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    assert len(OBC_boundary) == 12
    assert OBC_boundary[0] == (-26.5695640, 151.8373730)
    
def test_image_shape():
    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    assert image_shape(img) == (1280, 960)
    
def test_image_width():
    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    assert image_width(img) == 1280
    
def test_SubImage():
    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    region = (10, 10, 30, 30)
    subimage = SubImage(img, region)
    assert subimage.shape == (30, 30, 3)

    region = (1270, 950, 20, 20)
    subimage = SubImage(img, region)
    assert subimage.shape == (20, 20, 3)
    
def test_OverlayImage():
    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    region = (10, 10, 30, 30)
    subimage = SubImage(img, region)
    OverlayImage(img, subimage, 200, 150)
    assert subimage[0, 0, 1] == img[150, 200, 1]
    assert subimage[10, 10, 1] == img[160, 210, 1]
    assert subimage[20, 20, 1] == img[170, 220, 1]
    assert subimage[29, 29, 1] == img[179, 229, 1]
    
def test_SaturateImage():
    img = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    region = (10, 10, 30, 30)
    subimage = SubImage(img, region)
    newimage = SaturateImage(subimage, 2, 3)
    assert newimage.shape == (60, 60, 3)
    assert newimage[0,0,0] == 15
    assert newimage[0,0,1] == 12
    assert newimage[0,0,2] == 18

def test_set_system_clock():
    curtime = int(time.time())
    set_system_clock(curtime)
    assert time.time() - curtime < 1
    

