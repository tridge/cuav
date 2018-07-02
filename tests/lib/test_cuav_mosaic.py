#!/usr/bin/env python
'''
test program for cuav_mosiac
'''

import sys, os, time, random, functools, cv2
import pytest
import numpy as np
from cuav.lib import cuav_region, cuav_util
from cuav.lib import mav_position
from cuav.lib import cuav_mosaic
from cuav.camera.cam_params import CameraParams

#Python 2/3 compatibility
try:
    import mock
except ImportError:
    from unittest import mock

def test_MosaicRegion():
    regOne = cuav_region.Region(1020, 658, 1050, 678, (30, 30))
    regOne.latlon=(-26.6398870, 151.8220000)
    regOne.score = 20
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    pos.time = int(time.time())
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    full_thumbnail = im_orig[1:80, 1:80]
    small_thumbnail = im_orig[2:12, 10:20]
    mos = cuav_mosaic.MosaicRegion(2, regOne, "nofile.png", pos, full_thumbnail, small_thumbnail)
    mos.tag_image_available()
    assert mos.small_thumbnail[2,9,0] == 0
    assert mos.small_thumbnail[2,9,1] == 255
    assert mos.small_thumbnail[2,9,2] == 255
    assert "MavPosition(pos -30.000000 145.000000 alt=34.6 roll=20.0 pitch=-56.7 yaw=345.0)" in str(mos)

def test_MosaicImage():
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=None)
    pos.time = int(time.time())
    im = cuav_mosaic.MosaicImage(int(time.time()), "nofile.png", pos)
    assert "nofile.png" in str(im)
    
def test_ExtractThumbs():
    regions = []
    regions.append(cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20))
    regions.append(cuav_region.Region(30, 54, 50, 74, (20, 20), compactness=2.1, scan_score=15))
    regions.append(cuav_region.Region(60, 24, 170, 134, (110, 110), compactness=0, scan_score=0))
    im_orig = cv2.imread(os.path.join(os.getcwd(), 'tests', 'testdata', 'test-8bit.png'))
    composite = cuav_region.CompositeThumbnail(im_orig, regions)
    thumbs = cuav_mosaic.ExtractThumbs(composite, 3)
    assert len(thumbs) == 3
    assert cuav_util.image_shape(thumbs[0]) == (100,100)
    
def test_Mosaic():
    #slipmap = mp_slipmap.MPSlipMap(service='GoogleSat', elevation=True, title='Map')
    #mocked_slipmap.return_value = 1
    #monkeypatch.setattr('slipmap', lambda x: 1)
    mocked_slipmap = mock.MagicMock(return_value=1)
    C_params = CameraParams(lens=4.0, sensorwidth=5.0, xresolution=1280, yresolution=960)
    mosaic = cuav_mosaic.Mosaic(mocked_slipmap, C=C_params)
    mosaic.set_mosaic_size((200, 200))
    assert mosaic.mosaic.shape == (175, 175, 3)
    
    f = os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png')
    img = cv2.imread(f)
    pos = mav_position.MavPosition(-30, 145, 34.56, 20, -56.67, 345, frame_time=1478994408.76)
    regions = []
    regions.append(cuav_region.Region(1020, 658, 1050, 678, (30, 30), compactness=5.4, scan_score=20))
    regions.append(cuav_region.Region(30, 54, 50, 74, (20, 20), compactness=2.1, scan_score=15))
    regions.append(cuav_region.Region(30, 54, 55, 79, (25, 25), compactness=3.1, scan_score=10))
    for i in range(40):
        regions.append(cuav_region.Region(200, 600, 220, 620, (20, 20), compactness=2.2, scan_score=45))
    composite = cuav_region.CompositeThumbnail(img, regions)
    thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
    mosaic.add_regions(regions, thumbs, f, pos)
    mosaic.add_image(1478994408.76, f, pos)
    
    mosaic.show_region(0)

    mosaic.view_imagefile(f)
    
    
    assert mosaic.find_image_idx(f) == 0
    mosaic.view_imagefile_by_idx(0)
    
    mocked_key = mock.MagicMock(return_value=1)
    mocked_key.objkey = "region 1"
    assert mosaic.show_selected(mocked_key) == True
    
    mosaic.show_closest((-30, 145), mocked_key)
    
    mosaic.view_image.terminate()
    
    #mosaic.map_menu_callback
    
    #mosaic.map_callback
    
    OBC_boundary = cuav_util.polygon_load(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    mosaic.set_boundary(OBC_boundary)
    
    mosaic.change_page(1)
    mosaic.hide_page()
    assert len(mosaic.regions_sorted) == 25
    mosaic.unhide_all()
    assert len(mosaic.regions_sorted) == 43
    
    for i in ['Score', 'ScoreReverse', 'Compactness', 'Distinctiveness', 'Whiteness', 'Time']:
        mosaic.sort_type = i
        mosaic.re_sort()
    
    #mosaic.menu_event
    
    assert mosaic.started() == True
    
    mosaic.popup_show_image(mosaic.regions[2])
    
    mosaic.popup_fetch_image(mosaic.regions[2], 'fetchImageFull')
    
    assert len(mosaic.get_image_requests()) == 1
    
    mosaic.view_image.terminate()
    
    #mosaic.menu_event_view
    
    mocked_pos = mock.MagicMock(return_value=1)
    mocked_pos.x = 10
    mocked_pos.y = 10
    assert mosaic.pos_to_region(mocked_pos) == mosaic.regions[0]
    
    assert mosaic.objkey_to_region(mocked_key) == mosaic.regions[1]
    
    #mosaic.mouse_event
    
    #mosaic.mouse_event_view
    
    mosaic.key_event(1)
    
    assert mosaic.region_on_page(2, 0) == True
    assert mosaic.region_on_page(2000, 20) == False
    
    mosaic.mouse_region = mosaic.regions[0]
    mosaic.display_mosaic_region(0)
    
    mosaic.redisplay_mosaic()
    
    assert mosaic.make_thumb(img, regions[0], 8).shape == (8, 8, 3)
    assert mosaic.make_thumb(img, regions[0], 30).shape == (30, 30, 3)
    
    mosaic.tag_image(1478994408.76)
    
    #mosaic.check_events
    
    mosaic.image_mosaic.terminate()
    
