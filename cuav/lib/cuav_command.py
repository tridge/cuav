#!/usr/bin/env python
'''Commands sent between the GCS and UAV for camera control
and image transfer'''
import time

class StampedCommand:
    def __init__(self):
        self.timestamp = time.time()
        self.blockid = None

class ImagePacket(StampedCommand):
    '''a jpeg image sent to the ground station'''
    def __init__(self, frame_time, jpeg, pos, priority):
        StampedCommand.__init__(self)
        self.frame_time = frame_time
        self.jpeg = jpeg
        self.pos = pos
        self.priority = priority

class ThumbPacket(StampedCommand):
    '''a thumbnail region sent to the ground station'''
    def __init__(self, frame_time, regions, thumb, pos):
        StampedCommand.__init__(self)
        self.frame_time = frame_time
        self.regions = regions
        self.thumb = thumb
        self.pos = pos

class CommandPacket(StampedCommand):
    '''a command to run on the plane'''
    def __init__(self, command):
        StampedCommand.__init__(self)
        self.command = command

class CommandResponse(StampedCommand):
    '''a command response from the plane'''
    def __init__(self, response):
        StampedCommand.__init__(self)
        self.response = response
        
class ImageRequest(StampedCommand):
    '''request a jpeg image from the aircraft'''
    def __init__(self, frame_time, fullres):
        StampedCommand.__init__(self)
        self.frame_time = frame_time
        self.fullres = fullres

class HeartBeat(StampedCommand):
    '''generic heartbeat to keep bsend alive'''
    def __init__(self):
        StampedCommand.__init__(self)
        pass

class CameraMessage(StampedCommand):
    '''critical camera message'''
    def __init__(self, msg):
        StampedCommand.__init__(self)
        self.msg = msg

class ChangeCameraSetting(StampedCommand):
    '''update a camera setting'''
    def __init__(self, name, value):
        StampedCommand.__init__(self)
        self.name = name
        self.value = value

class ChangeImageSetting(StampedCommand):
    '''update a image setting'''
    def __init__(self, name, value):
        StampedCommand.__init__(self)
        self.name = name
        self.value = value

class BlockCancel(StampedCommand):
    '''cancel object for callback on send1 complete'''
    def __init__(self, blockid):
        StampedCommand.__init__(self)
        self.blockid = blockid
        

