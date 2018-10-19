#!/usr/bin/env python

'''Test cuav air module
'''

import sys
import pytest
import os
import mock
import threading, time

#for generating mocked mavlink messages
from pymavlink.dialects.v20 import common

import cuav.modules.camera_ground as camera_ground
import cuav.modules.camera_air as camera_air
from cuav.lib import cuav_command
from cuav.lib import cuav_util

@pytest.fixture
def mpstate():
    mpstatetmp = mock.Mock() #mpstatedummy.MPState()
    mpstatetmp.public_modules = {}
    mpstatetmp.command_map = {}
    mpstatetmp.completions = {}
    mpstatetmp.completion_functions = {}
    mpstatetmp.status.logdir = os.path.join(os.getcwd(), 'gnd')
    mpstatetmp.mav_master = [mock.Mock()]
    return mpstatetmp


def sim_camera():
    '''Create a thread that symlinks a bunch of test images in turns,
    with time delay'''
    t = threading.Thread(target=sim_camera_threadfunc)
    t.daemon = True
    t.start()
    return t

@pytest.fixture
def image_file():
    return str(os.path.join(os.getcwd(), 'tests', 'testdata', 'curcam.png'))

def sim_camera_threadfunc():
    '''Loops through all the images to simulate
    a series of camera captures'''
    images = [os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'),
                os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465160Z.png'),
                os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465213Z.png')]
    for image in images:
        try:
            os.unlink(image_file())
        except Exception:
            pass
        os.symlink(image, image_file())
        time.sleep(0.2)
    os.remove(image_file())


def test_load_module(mpstate):
    '''Just initialise the module'''
    loadedModule = camera_ground.init(mpstate)
    loadedModule.unload()

def test_module_settings(mpstate):
    '''try changing module settings via MAVProxy CLI'''
    loadedModule = camera_ground.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    assert loadedModule.camera_settings.camparms == parms

    loadedModule.cmd_camera(["set", "air_address", "127.0.0.1:15550:14550:45, 127.0.0.1:4500:1234:6000"])
    assert loadedModule.camera_settings.air_address == "127.0.0.1:15550:14550:45, 127.0.0.1:4500:1234:6000"

    loadedModule.unload()

def test_camera_commands(mpstate):
    '''Initialise the camera view frame and boundary'''
    loadedModule = camera_ground.init(mpstate)

    parms = "/data/ChameleonArecort/params.json"
    bnd = str(os.path.join(os.getcwd(), 'tests', 'testdata', 'OBC_boundary.txt'))
    loadedModule.cmd_camera(["set", "camparms", parms])

    loadedModule.cmd_camera(["view"])

    time.sleep(1.0)
    assert loadedModule.view_thread is not None

    loadedModule.cmd_camera(["boundary", bnd])

    loadedModule.unload()
    time.sleep(1.0)

    assert loadedModule.boundary_polygon is not None


def test_camera_thumbs(mpstate, image_file):
    '''Send some thumbnails to the view via the camera_air module'''
    loadedModuleAir = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModuleAir.cmd_camera(["set", "camparms", parms])
    loadedModuleAir.cmd_camera(["set", "imagefile", image_file])
    loadedModuleAir.cmd_camera(["set", "minscore", "0"])
    loadedModuleAir.cmd_camera(["set", "gcs_address", "127.0.0.1:14000:15000:45, 127.0.0.1:14500:15500:60000"])

    loadedModuleGround = camera_ground.init(mpstate)
    loadedModuleGround.cmd_camera(["set", "camparms", parms])
    loadedModuleGround.cmd_camera(["set", "air_address", "127.0.0.1:15000:14000:45, 127.0.0.1:15500:14500:60000"])
    loadedModuleGround.cmd_camera(["view"])

    #start the image stream and modules
    capture_thread = sim_camera()
    time.sleep(0.05)
    loadedModuleAir.cmd_camera(["start"])
    time.sleep(4.0)

    #and request a fullsize image
    filename = os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465160Z.png')
    frame_time = cuav_util.parse_frame_time(filename)
    loadedModuleGround.mosaic.tag_image(frame_time)
    time.sleep(2.0)

    loadedModuleAir.unload()
    loadedModuleGround.unload()
    capture_thread.join(1.0)

    #and do the asserts
    assert loadedModuleAir.region_count > 0
    assert loadedModuleGround.region_count > 0
    assert loadedModuleGround.thumb_count > 0
    assert loadedModuleGround.image_count > 0
