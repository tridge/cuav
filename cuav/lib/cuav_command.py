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

class PreviewPacket(StampedCommand):
    '''a jpeg image sent to the ground station for preview window'''
    def __init__(self, jpeg):
        StampedCommand.__init__(self)
        self.jpeg = jpeg
        
class FilePacket(StampedCommand):
    '''a general file send'''
    def __init__(self, filename, contents):
        StampedCommand.__init__(self)
        self.filename = filename
        self.contents = contents
        
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
    def __init__(self, icount):
        StampedCommand.__init__(self)
        self.icount = icount

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
        

class MavSocket:
    '''map block_xmit onto MAVLink data packets'''
    def __init__(self, master):
        self.master = master
        self.incoming = []

    def sendto(self, buf, dest):
        dbuf = [ord(x) for x in buf]
        dbuf.extend([0]*(96-len(dbuf)))
        if len(buf) <= 16:
            self.master.mav.data16_send(0, len(buf), dbuf)
        elif len(buf) <= 32:
            self.master.mav.data32_send(0, len(buf), dbuf)
        elif len(buf) <= 64:
            self.master.mav.data64_send(0, len(buf), dbuf)
        elif len(buf) <= 96:
            self.master.mav.data96_send(0, len(buf), dbuf)
        else:
            print("PACKET TOO LARGE %u" % len(dbuf))
            raise RuntimeError('packet too large %u' % len(dbuf))

    def recvfrom(self, size):
        if len(self.incoming) == 0:
            return ('', 'mavlink')
        m = self.incoming.pop(0)
        data = m.data[:m.len]
        s = ''.join([chr(x) for x in data])
        buf = bytes(s)
        return (buf, 'mavlink')
