#!/usr/bin/env python
'''
OBC 2020 cuav module
'''

import time
import threading
import sys
import os
import numpy
import pickle
import functools
import cv2
import pkg_resources
import io
import picamera
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
from pymavlink import mavutil

from cuav.lib import block_xmit, cuav_command
from cuav.lib import video_encode

class CameraAirModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CameraAirModule, self).__init__(mpstate, "camera_air", "cuav camera control (air)", public=True, multi_vehicle=True)

        self.running = False
        self.unload_event = threading.Event()
        self.unload_event.clear()
        self.capture_thread = None
        self.is_armed = True
        self.airstart_triggered = False
        self.transmit_queue = Queue.Queue()
        self.capture_count = 0
        self.encoder = video_encode.VideoWriter()
        self.camera = picamera.PiCamera()
        self.start_time = None
        self.handled_timestamps = {}

        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.camera_settings = MPSettings(
            [ MPSetting('quality', int, 90, 'Compression Quality', range=(1,100), increment=1, tab='GCS'),
              MPSetting('cropX', int, 0, 'crop X position', range=(0,2000), increment=1, tab='GCS'),
              MPSetting('cropY', int, 0, 'crop X position', range=(0,2000), increment=1, tab='GCS'),
              MPSetting('cropW', int, 0, 'crop width', range=(0,2000), increment=1, tab='GCS'),
              MPSetting('cropH', int, 0, 'crop height', range=(0,2000), increment=1, tab='GCS'),
              MPSetting('clock_sync', bool, False, 'GPS Clock Sync'),
              MPSetting('flipV', bool, False, 'flip vertically'),
              MPSetting('flipH', bool, False, 'flip horizontally'),
              MPSetting('save_images', bool, False, 'save images'),
              MPSetting('min_width', int, 32, 'min delta width'),
              MPSetting('m_bandwidth', int, 2000, 'max bandwidth on mavlink', increment=1, tab='GCS'),
              MPSetting('m_maxqueue', int, 20, 'Maximum images queue for mavlink', tab='GCS'),
              MPSetting('minspeed', int, 20, 'For airstart, minimum speed for capture to start'),
              MPSetting('minalt', int, 30, 'MinAltitude of images', range=(0,10000), increment=1),
              ],
            title='Camera Settings'
            )

        self.msocket = None
        self.msend = None
        self.last_heartbeat = time.time()

        self.add_command('camera', self.cmd_camera,
                         'camera control',
                         ['<start|stop|status>',
                          'set (CAMERASETTING)'])
        self.add_completion_function('(CAMERASETTING)', self.camera_settings.completion)

        # prevent loopback of messages
        for mtype in ['DATA16', 'DATA32', 'DATA64', 'DATA96']:
            self.module('link').no_fwd_types.add(mtype)
        
        print("camera initialised")

    def cmd_camera(self, args):
        '''camera commands'''
        usage = "usage: camera <start|airstart|stop|status|queue|set>"
        if len(args) == 0:
            print(usage)
            return
        if args[0] == "start":
            if not self.running:
                self.encoder.set_crop((self.camera_settings.cropX,
                                       self.camera_settings.cropY,
                                       self.camera_settings.cropW,
                                       self.camera_settings.cropH))
                self.capture_thread = self.start_thread(self.capture_threadfunc)
                self.transmit_thread = self.start_thread(self.transmit_threadfunc)
                time.sleep(0.1)
                self.running = True
                self.send_message("Started cuav running")
                print("Started cuav running")
            else:
                self.send_message("cuav already running")
                print("cuav already running")
        elif args[0] == "stop":
            self.running = False
            self.start_time = None
            print("Stopped cuav")
            self.send_message("Stopped cuav")
        elif args[0] == "status":
            print("status....")
        elif args[0] == "set":
            self.camera_settings.command(args[1:])
        else:
            print(usage)

    def cap_image_CV(self):
        '''capture one image'''
        s = io.BytesIO()
        self.camera.capture(s, "jpeg")
        s.seek(0)
        data = numpy.fromstring(s.getvalue(), dtype=numpy.uint8)
        img = cv2.imdecode(data, 1)
        if self.camera_settings.flipV:
            img = cv2.flip(img, 0)[:,:]
        if self.camera_settings.flipH:
            img = cv2.flip(img, 1)[:,:]
        if self.camera_settings.save_images:
            cv2.imwrite("img%u.jpg" % self.capture_count, img)
        return img
            
    def capture_threadfunc(self):
        '''image capture thread'''
        last_t = time.time()
        while True:
            if not self.running:
                self.encoder.reset()
                time.sleep(0.1)
                continue
            if self.is_armed:
                self.encoder.reset()
                self.capture_count = 0
                time.sleep(1)
                continue
            target_t = last_t + 0.95
            now = time.time()
            if now < target_t:
                time.sleep(target_t - now)
            img = self.cap_image_CV()
            now = time.time()
            if self.start_time is None:
                self.start_time = now
            tstamp_ms = int((now - self.start_time)*1000)
            self.encoder.min_width = self.camera_settings.min_width
            self.encoder.quality = self.camera_settings.quality
            enc = self.encoder.add_image(img, tstamp_ms)
            self.encoder.report()
            if len(enc) == 0:
                continue
            if self.capture_count == 0:
                priority = 10000
            else:
                priority = 9000
            pkt = cuav_command.ImageDelta(tstamp_ms, enc, priority)
            if self.msend:
                self.transmit_queue.put((pkt, priority, self.msend))
            self.capture_count += 1
        
    def transmit_threadfunc(self):
        '''thread for image and message transmit to camera_ground
        in addition to reading commands from the camera_ground'''
        self.start_aircraft_bsend()
        self.spacewarning = False

        while (not self.unload_event.wait(0.05)) or self.airstart_triggered:
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

    def start_aircraft_bsend(self):
        '''start bsend for aircraft side'''
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
        if self.capture_thread is not None:
            self.capture_thread.join(1.0)
            self.scan_thread.join(1.0)
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

        if isinstance(obj, cuav_command.ImageRequest):
            self.handle_image_request(obj, bsend)

        if isinstance(obj, cuav_command.ChangeCameraSetting):
            self.camera_settings.set(obj.name, obj.value)
            self.camera_settings_callback(obj)

        if isinstance(obj, cuav_command.ChangeImageSetting):
            self.image_settings.set(obj.name, obj.value)
            self.image_settings_callback(obj)

        if isinstance(obj, cuav_command.CommandPacket):
            self.handle_command_packet(obj, bsend)

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        if m.get_type() in [ 'DATA16', 'DATA32', 'DATA64', 'DATA96' ]:
            if self.msocket is not None:
                self.msocket.incoming.append(m)                
        if self.mpstate.status.watch in ["camera","queue"] and time.time() > self.last_watch+1:
            self.last_watch = time.time()
            self.cmd_camera(["status" if self.mpstate.status.watch == "camera" else "queue"])
        # update position interpolator
        if m.get_type() == 'SYSTEM_TIME' and self.camera_settings.clock_sync and self.capture_thread is not None:
            # optionally sync system clock on the capture side
            self.sync_gps_clock(m.time_unix_usec)
        if m.get_type() == 'VFR_HUD' and self.airstart_triggered and not self.running:
            #if the airstart is triggered and we're flying, then start capture
            if m.airspeed > self.camera_settings.minspeed or m.groundspeed > self.camera_settings.minspeed:
                self.running = True
                self.capture_thread = self.start_thread(self.capture_threadfunc)
                self.transmit_thread = self.start_thread(self.transmit_threadfunc)
                self.send_message("Started cuav running")
                print("Started cuav running")
        if m.get_type() == "HEARTBEAT" and m.type != mavutil.mavlink.MAV_TYPE_GCS:
            was_armed = self.is_armed
            self.is_armed = (m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

    def sync_gps_clock(self, time_usec):
        '''sync system clock with GPS time'''
        if time_usec == 0:
            # no GPS lock
            return
        if os.geteuid() != 0:
            # can only do this as root
            return
        time_seconds = time_usec*1.0e-6
        if self.have_set_gps_time and abs(time_seconds - time.time()) < 10:
            # only change a 2nd time if time is off by 10 seconds
            return
        t1 = time.time()
        cuav_util.set_system_clock(time_seconds)
        t2 = time.time()
        print("Changed system time by %.2f seconds" % (t2-t1))
        self.have_set_gps_time = True

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

def init(mpstate):
    '''initialise module'''
    return CameraAirModule(mpstate)
