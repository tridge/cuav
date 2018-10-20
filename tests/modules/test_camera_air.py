#!/usr/bin/env python

'''Test cuav air module
'''

import sys
import pytest
import os
import threading, time, pickle

if sys.version_info >= (3, 3):
    from unittest import mock
else:
    import mock

#for generating mocked mavlink messages
from pymavlink.dialects.v20 import common

import cuav.modules.camera_air as camera_air
from cuav.lib import block_xmit, cuav_command, cuav_util

@pytest.fixture
def mpstate():
    mpstatetmp = mock.Mock() #mpstatedummy.MPState()
    mpstatetmp.public_modules = {}
    mpstatetmp.command_map = {}
    mpstatetmp.completions = {}
    mpstatetmp.completion_functions = {}
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
    loadedModule = camera_air.init(mpstate)
    loadedModule.unload()

def test_module_settings(mpstate):
    '''try changing module settings via MAVProxy CLI'''
    loadedModule = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    assert loadedModule.camera_settings.camparms == parms

    image = str(os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465120Z.png'))
    loadedModule.cmd_camera(["set", "imagefile", image])
    assert loadedModule.camera_settings.imagefile == image

    loadedModule.cmd_camera(["set", "gcs_address", "127.0.0.1:14550:45, 127.0.0.1:1234:6000"])
    assert loadedModule.camera_settings.gcs_address == "127.0.0.1:14550:45, 127.0.0.1:1234:6000"

    loadedModule.unload()

def test_camera_start(mpstate, image_file):
    '''put a few images through the module and check they come
    out via the block xmit'''
    loadedModule = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    loadedModule.cmd_camera(["set", "imagefile", image_file])
    loadedModule.cmd_camera(["set", "minscore", "0"])

    loadedModule.cmd_camera(["set", "gcs_address", "127.0.0.1:14550:14560:9000, 127.0.0.1:14650:14660:15000"])

    #Set up the ground side recievers
    b1 = block_xmit.BlockSender(dest_ip='127.0.0.1', port = 14550, dest_port = 14560)
    b2 = block_xmit.BlockSender(dest_ip='127.0.0.1', port = 14650, dest_port = 14660)
    blk1 = None
    blk2 = None

    b2.tick()
    b1.tick()

    capture_thread = sim_camera()
    time.sleep(0.05)
    loadedModule.cmd_camera(["start"])
    time.sleep(0.8)
    loadedModule.cmd_camera(["status"])
    loadedModule.cmd_camera(["queue"])
    #get the sent data
    b2.tick()
    b1.tick()
    blk1 = pickle.loads(str(b1.recv(0.1)))
    blk2 = pickle.loads(str(b2.recv(0.1)))
    time.sleep(0.05)
    loadedModule.cmd_camera(["status"])
    loadedModule.cmd_camera(["stop"])
    loadedModule.unload()
    capture_thread.join(1.0)

    assert loadedModule.capture_count == 3
    assert loadedModule.scan_count == 3
    assert loadedModule.region_count > 0
    assert isinstance(blk1, cuav_command.StampedCommand)
    assert isinstance(blk2, cuav_command.StampedCommand)
    assert abs(blk1.timestamp - blk2.timestamp) < 0.01
    #assert loadedModule.xmit_queue == [0, 0]

def test_camera_command(mpstate, image_file):
    '''send some commands via the block_xmit'''
    loadedModule = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    loadedModule.cmd_camera(["set", "imagefile", image_file])
    loadedModule.cmd_camera(["set", "minscore", "0"])
    loadedModule.cmd_camera(["set", "gcs_address", "127.0.0.1:14550:14560:9000"])

    pkt = cuav_command.ChangeCameraSetting("minscore", 50)
    b1 = block_xmit.BlockSender(dest_ip='127.0.0.1', port = 14550, dest_port = 14560)
    buf = pickle.dumps(pkt)

    loadedModule.cmd_camera(["start"])
    time.sleep(0.1)
    b1.tick()
    b1.send(buf)
    b1.tick()
    time.sleep(0.1)
    loadedModule.cmd_camera(["stop"])
    loadedModule.unload()

    assert loadedModule.camera_settings.minscore == 50

def test_camera_image_request(mpstate, image_file):
    '''image request via the block_xmit'''
    loadedModule = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    loadedModule.cmd_camera(["set", "imagefile", image_file])
    loadedModule.cmd_camera(["set", "minscore", "0"])
    loadedModule.cmd_camera(["set", "gcs_address", "127.0.0.1:14550:14560:2000000"])

    filename = os.path.join(os.getcwd(), 'tests', 'testdata', 'raw2016111223465160Z.png')
    pkt = cuav_command.ImageRequest(cuav_util.parse_frame_time(filename), True)
    buf = pickle.dumps(pkt)

    capture_thread = sim_camera()
    time.sleep(0.05)
    loadedModule.cmd_camera(["start"])
    time.sleep(0.8)
    loadedModule.cmd_camera(["status"])

    #don't load the block xmit until afterwards
    b1 = block_xmit.BlockSender(dest_ip='127.0.0.1', port = 14550, dest_port = 14560)
    b1.tick()
    b1.send(buf)
    #b1.tick()
    time.sleep(0.1)
    blkret = []
    while True:
        try:
            b1.tick()
            blk = pickle.loads(str(b1.recv(0.01, True)))
            #only want paricular packets - discard all the heartbeats, etc
            if isinstance(blk, cuav_command.ImagePacket):
                blkret.append(blk)
                break
            time.sleep(0.05)
        except pickle.UnpicklingError:
            continue

    loadedModule.cmd_camera(["stop"])
    loadedModule.unload()
    capture_thread.join(1.0)

    assert len(blkret) == 1
    #not sure if the last or 2nd last packed will contain the image - depends on the exact thread timing
    assert isinstance(blkret[0], cuav_command.ImagePacket)
    assert blkret[0].jpeg is not None

def test_camera_airstart(mpstate, image_file):
    '''test the airstart - that cauv auto starts after vehicle has taken off'''
    loadedModule = camera_air.init(mpstate)
    parms = "/data/ChameleonArecort/params.json"
    loadedModule.cmd_camera(["set", "camparms", parms])
    loadedModule.cmd_camera(["set", "imagefile", image_file])
    loadedModule.cmd_camera(["set", "minspeed", "20"])
    loadedModule.cmd_camera(["set", "gcs_address", "127.0.0.1:14550:14560:2000000"])

    #load the other side of the xmit link
    b1 = block_xmit.BlockSender(dest_ip='127.0.0.1', port = 14550, dest_port = 14560)
    blkret = []
    b1.tick()

    loadedModule.cmd_camera(["airstart"])
    #send a mavlink packet of VFR_HUD that the airspeed is above the threshold
    #See "common.py" in pymavlink dialects for message classes
    msg = common.MAVLink_vfr_hud_message(airspeed=21, groundspeed=15, heading=90, throttle=80, alt=30, climb=3.2)
    msg._timestamp = time.time()
    
    loadedModule.mavlink_packet(msg)
    #get the packets
    time.sleep(0.1)
    while True:
        try:
            b1.tick()
            blk = pickle.loads(b1.recv(0.01, True))
            if isinstance(blk, cuav_command.CommandResponse) or isinstance(blk, cuav_command.CameraMessage):
                blkret.append(blk)
            time.sleep(0.05)
        except pickle.UnpicklingError:
            break
    loadedModule.cmd_camera(["stop"])
    loadedModule.unload()

    assert len(blkret) == 2
    assert blkret[0].msg == "cuav airstart ready"
    assert blkret[1].msg == "Started cuav running"

