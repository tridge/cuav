#!/usr/bin/env python
'''
OBC 2020 cuav module
'''

import time
import threading
import sys
import os
import io
import numpy
import pickle
import functools
import cv2
import pkg_resources
import io
import Queue

try:
    # py2
    from StringIO import StringIO
except ImportError:
    # py3
    from io import StringIO

from MAVProxy.modules.lib import mp_module
from MAVProxy.modules.lib import multiproc
from MAVProxy.modules.lib import mp_settings
from MAVProxy.modules.lib import mp_image
from pymavlink import mavutil

from cuav.lib import block_xmit, cuav_command
from cuav.lib import video_play

class CameraGroundModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CameraGroundModule, self).__init__(mpstate, "camera_ground", "cuav camera control (ground)", public=True, multi_vehicle=True)

        self.unload_event = threading.Event()
        self.unload_event.clear()
        self.transmit_queue = Queue.Queue()
        self.decoder = video_play.VideoReader()
        self.is_armed = False
        self.capture_count = 0
        self.image = None
        self.last_capture_count = None
        self.handled_timestamps = {}
        self.viewer = mp_image.MPImage(title='Image', width=200, height=200, auto_size=True)

        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.camera_settings = MPSettings(
            [ 
              MPSetting('m_bandwidth', int, 500, 'max bandwidth on mavlink', increment=1, tab='GCS'),
              MPSetting('m_maxqueue', int, 5, 'Maximum images queue for mavlink', tab='GCS'),
              ],
            title='Camera Settings'
            )

        self.msocket = None
        self.msend = None
        self.last_heartbeat = time.time()
        self.transmit_thread = self.start_thread(self.transmit_threadfunc)

        self.add_command('camera', self.cmd_camera,
                         'camera control',
                         ['<status>',
                          'set (CAMERASETTING)'])
        self.add_completion_function('(CAMERASETTING)', self.camera_settings.completion)

        # prevent loopback of messages
        for mtype in ['DATA16', 'DATA32', 'DATA64', 'DATA96']:
            self.module('link').no_fwd_types.add(mtype)
        
        print("camera ground initialised")

    def cmd_camera(self, args):
        '''camera commands'''
        usage = "usage: camera <status|queue|set>"
        if len(args) == 0:
            print(usage)
            return
        if args[0] == "status":
            print("status....")
        elif args[0] == "set":
            self.camera_settings.command(args[1:])
        else:
            print(usage)

    def transmit_threadfunc(self):
        '''thread for image and message transmit to camera_ground
        in addition to reading commands from the camera_ground'''
        self.start_bsend()
        self.spacewarning = False

        while not self.unload_event.wait(0.05):
            if self.msend is not None:
                self.msend.tick(packet_count=1000, max_queue=self.camera_settings.m_maxqueue)
                self.check_commands(self.msend)
            self.send_heartbeats()

            while not self.transmit_queue.empty():
                (pkt, priority, linktosend) = self.transmit_queue.get()
                if self.msend:
                    self.send_object(pkt, priority, self.msend)

            #update the stats
            self.xmit_queue = []
            self.efficiency = []
            self.bandwidth_used = []
            self.rtt_estimate = []
            if self.msend is not None:
                self.xmit_queue.append(self.msend.sendq_size())
                self.efficiency.append(self.msend.get_efficiency())
                self.bandwidth_used.append(self.msend.get_bandwidth_used())
                self.rtt_estimate.append(self.msend.get_rtt_estimate())
                
    def send_heartbeats(self):
        '''possibly send heartbeat msgs'''
        now = time.time()
        if now - self.last_heartbeat > 5:
            self.last_heartbeat = now
            self.send_heartbeat()

    def start_bsend(self):
        '''start bsend'''
        if self.msend is None:
            self.msocket = cuav_command.MavSocket(self.mpstate.mav_master[0])
            self.msend = block_xmit.BlockSender(mss=96, sock=self.msocket, dest_ip='mavlink', dest_port=0, backlog=5, debug=False)
            self.msend.set_bandwidth(self.camera_settings.m_bandwidth)

    def start_thread(self, fn):
        '''start a thread running'''
        t = threading.Thread(target=fn)
        t.daemon = True
        t.start()
        return t

    def unload(self):
        '''unload module'''
        self.running = False
        self.unload_event.set()
        if self.transmit_thread is not None:
            self.transmit_thread.join(1.0)
        print('camera unload OK')

    def check_commands(self, bsend):
        '''check for remote commands'''
        if bsend is None:
            return
        buf = bsend.recv(0)
        if buf is None:
            return
        try:
            obj = pickle.loads(buf)
            if obj == None:
                return
        except Exception as e:
            return

        if isinstance(obj, cuav_command.StampedCommand):
            if obj.timestamp in self.handled_timestamps:
                # we've seen this packet before, discard
                return
            self.handled_timestamps[obj.timestamp] = time.time()

        if isinstance(obj, cuav_command.ImageDelta):
            self.handle_image_delta(obj, bsend)

        if isinstance(obj, cuav_command.CommandPacket):
            self.handle_command_packet(obj, bsend)

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        if self.mpstate.status.watch in ["camera","queue"] and time.time() > self.last_watch+1:
            self.last_watch = time.time()
            self.cmd_camera(["status" if self.mpstate.status.watch == "camera" else "queue"])
        if m.get_type() == "HEARTBEAT" and m.type != mavutil.mavlink.MAV_TYPE_GCS:
            self.is_armed = (m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        if m.get_type() in [ 'DATA16', 'DATA32', 'DATA64', 'DATA96' ]:
            if self.msocket is not None:
                self.msocket.incoming.append(m)                

    def send_heartbeat(self):
        '''send a heartbeat'''
        pkt = cuav_command.HeartBeat(self.capture_count)
        if self.msend is not None:
            self.transmit_queue.put((pkt, None, 'msend'))

    def send_message(self, msg):
        '''send a message'''
        pkt = cuav_command.CameraMessage(msg)
        if self.msend is not None:
            self.transmit_queue.put((pkt, 100, 'msend'))

    def send_object_complete(self, obj, bsend):
        '''called on complete of an send_object, cancelling send on other links'''
        if obj.blockid is not None:
            if self.msend is not None:
                self.msend.cancel(obj.blockid)

    def send_object(self, obj, priority=None, linktosend=None):
        '''send an object to all links if linktosend is none
        otherwise just send to the specified link'''
        try:
            buf = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("dump failed: ", ex)
            return
        if priority is None:
            priority = 10000
        #only send if the queue is not clogged
        if not self.msend:
            return
        obj.blockid = self.msend.send(buf, priority=priority, callback=functools.partial(self.send_object_complete, obj, self.msend))

    def handle_command_packet(self, obj, bsend):
        '''handle CommandPacket from other end'''
        stdout_saved = sys.stdout
        buf = StringIO()
        sys.stdout = buf
        self.mpstate.functions.process_stdin(obj.command, immediate=True)
        sys.stdout = stdout_saved
        if str(buf.getvalue().strip()):
            pkt = cuav_command.CommandResponse(str(buf.getvalue()).strip())
            self.transmit_queue.put((pkt, None, self.msend))

    def handle_image_delta(self, obj, bsend):
        '''handle a ImageDelta packet'''
        if self.capture_count == 0:
            if obj.priority < 10000:
                print("Skipping early image")
                return
        s = io.BytesIO()
        s.write(obj.delta)
        s.seek(0)
        (img,dt) = self.decoder.get_image(s)
        if img is None:
            return
        if self.capture_count == 0:
            print("Got first image: ", img.shape)
        self.capture_count += 1
        self.viewer.set_image(img)

    def idle_task(self):
        '''idle time handler'''
        pass
        

def init(mpstate):
    '''initialise module'''
    return CameraGroundModule(mpstate)
