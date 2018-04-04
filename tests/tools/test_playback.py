#!/usr/bin/env python

'''Test playback
'''

import sys
import pytest
import os
import cuav.tools.playback as playback
from cuav.lib import cuav_util

def test_do_playback():
    #Check if we're running under Windows:
    if sys.platform.startswith('win'):
        print("This script is not compatible with Windows")
        return
    
    logfile = os.path.join('.', 'tests', 'testdata', 'flight.tlog')
    outfile = os.path.join('.', 'cur_camera.png')
    images = []
    image_a = os.path.join('.', 'tests', 'testdata', 'raw2016111223465120Z.png')
    images.append(playback.ImageFile(cuav_util.parse_frame_time(image_a), image_a))
    image_b = os.path.join('.', 'tests', 'testdata', 'raw2016111223465160Z.png')
    images.append(playback.ImageFile(cuav_util.parse_frame_time(image_b), image_b))
    image_c = os.path.join('.', 'tests', 'testdata', 'raw2016111223465213Z.png')
    images.append(playback.ImageFile(cuav_util.parse_frame_time(image_c), image_c))
    playback.playback(os.path.join('.', 'tests', 'testdata', 'flight.tlog'), images, "127.0.0.1:14550", 57600, None, 10)
    assert os.path.isfile(outfile)
    os.remove(outfile)


