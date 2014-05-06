#!/usr/bin/env python
'''camera control for ptgrey chameleon camera'''

import time, threading, sys, os, numpy, Queue, errno, cPickle, signal, struct, fcntl, select, cStringIO
try:
    import cv2.cv as cv
except ImportError:
    import cv

from MAVProxy.modules.lib import mp_module

from cuav.image import scanner
from pymavlink import mavutil
from cuav.lib import cuav_mosaic, mav_position, cuav_util, cuav_joe, block_xmit, cuav_region
from MAVProxy.modules.lib import mp_settings
from MAVProxy.modules.lib import mp_image
from cuav.camera.cam_params import CameraParams
from MAVProxy.modules.mavproxy_map import mp_slipmap

# allow for replaying of previous flights
if os.getenv('FAKE_CHAMELEON'):
    print("Loaded fake chameleon backend")
    import cuav.camera.fake_chameleon as chameleon
else:
    import cuav.camera.chameleon as chameleon

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

class ImagePacket:
    '''a jpeg image sent to the ground station'''
    def __init__(self, frame_time, jpeg, xmit_queue, pos, priority):
        self.frame_time = frame_time
        self.jpeg = jpeg
        self.xmit_queue = xmit_queue
        self.pos = pos
        self.priority = priority

class ThumbPacket:
    '''a thumbnail region sent to the ground station'''
    def __init__(self, frame_time, regions, thumb, frame_loss, xmit_queue, pos):
        self.frame_time = frame_time
        self.regions = regions
        self.thumb = thumb
        self.frame_loss = frame_loss
        self.xmit_queue = xmit_queue
        self.pos = pos

class CommandPacket:
    '''a command to run on the plane'''
    def __init__(self, command):
        self.command = command

class CommandResponse:
    '''a command response from the plane'''
    def __init__(self, response):
        self.response = response


class ImageRequest:
    '''request a jpeg image from the aircraft'''
    def __init__(self, frame_time, fullres):
        self.frame_time = frame_time
        self.fullres = fullres

class ChangeCameraSetting:
    '''update a camera setting'''
    def __init__(self, name, value):
        self.name = name
        self.value = value

class ChangeImageSetting:
    '''update a image setting'''
    def __init__(self, name, value):
        self.name = name
        self.value = value
        

class CameraModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CameraModule, self).__init__(mpstate, "camera", "cuav camera control")

        self.running = False
        self.unload_event = threading.Event()
        self.unload_event.clear()

        self.capture_thread_h = None
        self.save_thread_h = None
        self.scan_thread1_h = None
        self.scan_thread2_h = None
        self.transmit_thread_h = None
        self.view_thread_h = None

        from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
        self.camera_settings = MPSettings(
            [ MPSetting('depth', int, 8, 'Image Depth', choice=['8', '16'], tab='Capture'),
              MPSetting('save_pgm', bool, True, 'Save Raw Images'),
              MPSetting('capture_brightness', int, 150, 'Capture Brightness', range=(10, 300), increment=1),
              MPSetting('gamma', int, 950, 'Capture Gamma', range=(0,1000), increment=1),
              MPSetting('roll_stabilised', bool, True, 'Roll Stabilised'),
              MPSetting('altitude', int, 0, 'Altitude', range=(0,10000), increment=1),
              MPSetting('filter_type', str, 'simple', 'Filter Type',
                        choice=['simple', 'compactness']),
              MPSetting('fullres', bool, False, 'Full Resolution'),
              MPSetting('framerate', str, 7, 'Frame Rate', choice=['1', '3', '7', '15']),
              MPSetting('process_divider', int, 1, 'Process Divider', range=(1,50), increment=1),
              MPSetting('use_capture_time', bool, False, 'Use Capture Time'),

              MPSetting('gcs_address', str, None, 'GCS Address', tab='GCS'),
              MPSetting('gcs_view_port', int, 7543, 'GCS View Port', range=(1, 30000), increment=1),
              MPSetting('gcs_slave', str, None, 'GCS Slave'),
              
              MPSetting('bandwidth',  int, 40000, 'Link1 Bandwdith', 'Comms'),
              MPSetting('bandwidth2', int, 2000, 'Link2 Bandwidth'),
              MPSetting('quality', int, 75, 'Compression Quality', range=(1,100), increment=1),
              MPSetting('transmit', bool, True, 'Transmit Enable'),
              MPSetting('send1', bool, True, 'Send on Link1'),
              MPSetting('send2', bool, True, 'Send on Link2'),
              MPSetting('maxqueue1', int, None, 'Maximum queue Link1'),
              MPSetting('maxqueue2', int, 30, 'Maxqueue queue Link2'),
              MPSetting('thumbsize', int, 60, 'Thumbnail Size', range=(10, 200), increment=1),
              MPSetting('mosaic_thumbsize', int, 35, 'Mosaic Thumbnail Size', range=(10, 200), increment=1),
              MPSetting('use_bsend2', bool, True, 'Enable Link2'),

              MPSetting('minscore', int, 75, 'Min Score Link1', range=(0,1000), increment=1, tab='Scoring'),
              MPSetting('minscore2', int, 500, 'Min Score Link2', range=(0,1000), increment=1),
              MPSetting('packet_loss', int, 0, 'Packet Loss', range=(0,100), increment=1, tab='Misc'),             
              MPSetting('clock_sync', bool, False, 'GPS Clock Sync'),             

              MPSetting('brightness', float, 1.0, 'Display Brightness', range=(0.1, 10), increment=0.1,
                        digits=2, tab='Display')
              ],
            title='Camera Settings'
            )

        self.image_settings = MPSettings(
            [ MPSetting('MinRegionArea', float, 0.15, range=(0,100), increment=0.05, digits=2, tab='Image Processing'),
              MPSetting('MaxRegionArea', float, 2.0, range=(0,100), increment=0.1, digits=1),
              MPSetting('MinRegionSize', float, 0.1, range=(0,100), increment=0.05, digits=2),
              MPSetting('MaxRegionSize', float, 2, range=(0,100), increment=0.1, digits=1),
              MPSetting('MaxRarityPct',  float, 0.02, range=(0,100), increment=0.01, digits=2),
              MPSetting('RegionMergeSize', float, 3.0, range=(0,100), increment=0.1, digits=1),
              MPSetting('SaveIntermediate', bool, False)
              ],
            title='Image Settings')

        self.capture_count = 0
        self.process_counter = 0
        self.scan_count = 0
        self.error_count = 0
        self.error_msg = None
        self.region_count = 0
        self.fps = 0
        self.scan_fps = 0
        self.cam = None
        self.save_queue = Queue.Queue()
        self.scan_queue = Queue.Queue()
        self.transmit_queue = Queue.Queue()
        self.viewing = False
        self.have_set_gps_time = False
        
        self.c_params = CameraParams(lens=4.0)
        self.jpeg_size = 0
        self.xmit_queue = 0
        self.xmit_queue2 = 0
        self.efficiency = 1.0

        self.last_watch = 0
        self.frame_loss = 0
        self.boundary = None
        self.boundary_polygon = None

        self.bandwidth_used = 0
        self.rtt_estimate = 0
        self.bsocket = None
        self.bsend = None
        self.bsend2 = None
        self.bsend_slave = None
        self.framerate = 0
        
        # setup directory for images
        self.camera_dir = os.path.join(self.logdir, "camera")
        cuav_util.mkdir_p(self.camera_dir)

        self.mpos = mav_position.MavInterpolator(backlog=5000, gps_lag=0.3)
        self.joelog = cuav_joe.JoeLog(os.path.join(self.camera_dir, 'joe.log'), append=self.continue_mode)
        # load camera params
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..',
                            'cuav', 'data', 'chameleon1_arecont0.json')
        if os.path.exists(path):
            self.c_params.load(path)
        else:
            print("Warning: %s not found" % path)

        self.add_command('camera', self.cmd_camera,
                         'camera control',
                         ['<start|stop|status|view|noview|boundary>',
                          'set (CAMERASETTING)'])
        self.add_completion_function('(CAMERASETTING)', self.settings.completion)
        self.add_command('remote', self.cmd_remote, "remote command", ['(COMMAND)'])
        self.add_completion_function('(CAMERASETTING)', self.camera_settings.completion)
        print("camera initialised")

    def cmd_camera(self, args):
        '''camera commands'''
        usage = "usage: camera <start|stop|status|view|noview|boundary|set|image>"
        if len(args) == 0:
            print(usage)
            return
        if args[0] == "start":
            self.capture_count = 0
            self.error_count = 0
            self.error_msg = None
            self.running = True
            if self.capture_thread_h is None:
                self.capture_thread_h = self.start_thread(self.capture_thread)
                self.save_thread_h = self.start_thread(self.save_thread)
                self.scan_thread1_h = self.start_thread(self.scan_thread)
                self.scan_thread2_h = self.start_thread(self.scan_thread)
                self.transmit_thread_h = self.start_thread(self.transmit_thread)
            print("started camera running")
        elif args[0] == "stop":
            self.running = False
            print("stopped camera capture")
        elif args[0] == "status":
            print("Cap imgs:%u err:%u scan:%u fr:%.3f regions:%u jsize:%.0f xmitq:%u/%u lst:%u sq:%.1f eff:%.2f" % (
                self.capture_count, self.error_count, self.scan_count, 
                self.framerate,
                self.region_count, 
                self.jpeg_size,
                self.xmit_queue, self.xmit_queue2, self.frame_loss, self.scan_queue.qsize(), self.efficiency))
            print("self.bsend2 is ", self.bsend2)
            if self.bsend2 is not None:
                self.bsend2.report(detailed=True)
        elif args[0] == "queue":
            print("scan %u  save %u  transmit %u  eff %.1f  bw %.1f  rtt %.1f" % (
                self.scan_queue.qsize(),
                self.save_queue.qsize(),
                self.transmit_queue.qsize(),
                self.efficiency,
                self.bandwidth_used,
                self.rtt_estimate))
        elif args[0] == "view":
            if self.mpstate.map is None:
                print("Please load map module first")
                return
            if not self.viewing:
                print("Starting image viewer")
            if self.view_thread_h is None:
                self.view_thread_h = self.start_thread(self.view_thread)
            self.viewing = True
        elif args[0] == "noview":
            if self.viewing:
                print("Stopping image viewer")
            self.viewing = False
        elif args[0] == "set":
            self.camera_settings.command(args[1:])
        elif args[0] == "image":
            self.image_settings.command(args[1:])
        elif args[0] == "boundary":
            if len(args) != 2:
                print("boundary=%s" % self.boundary)
            else:
                self.boundary = args[1]
                self.boundary_polygon = cuav_util.polygon_load(self.boundary)
                if self.mpstate.map is not None:
                    self.mpstate.map.add_object(mp_slipmap.SlipPolygon('boundary', self.boundary_polygon,
                                                                       layer=1, linewidth=2, colour=(0,0,255)))                
        else:
            print(usage)


    def cmd_remote(self, args):
        '''camera commands'''
        cmd = " ".join(args)
        pkt = CommandPacket(cmd)
        self.send_packet(pkt)

    def get_base_time(self):
        '''we need to get a baseline time from the camera. To do that we trigger
        in single shot mode until we get a good image, and use the time we 
        triggered as the base time'''
        frame_time = None
        error_count = 0

        print('Opening camera')
        h = chameleon.open(1, self.camera_settings.depth, self.camera_settings.capture_brightness)

        print('Getting camare base_time')
        while frame_time is None:
            try:
                im = numpy.zeros((960,1280),dtype='uint8' if self.camera_settings.depth==8 else 'uint16')
                base_time = time.time()
                chameleon.trigger(h, False)
                frame_time, frame_counter, shutter = chameleon.capture(h, 1000, im)
                base_time -= frame_time
            except chameleon.error:
                print('failed to capture')
                error_count += 1
            if error_count > 3:
                error_count = 0
                print('re-opening camera')
                chameleon.close(h)
                h = chameleon.open(1, self.camera_settings.depth, self.camera_settings.capture_brightness)
        print('base_time=%f' % base_time)
        return h, base_time, frame_time

    def capture_thread(self):
        '''camera capture thread'''
        print("running capture_thread")
        t1 = time.time()
        last_frame_counter = 0
        h = None
        last_gamma = 0
        last_framerate = 7

        raw_dir = os.path.join(self.camera_dir, "raw")
        cuav_util.mkdir_p(raw_dir)

        if self.continue_mode:
            mode = 'a'
        else:
            mode = 'w'
        gammalog = open(os.path.join(self.camera_dir, "gamma.log"), mode=mode)

        while not self.unload_event.wait(0.02):
            if not self.running:            
                if h is not None:
                    chameleon.close(h)
                    h = None
                continue

            try:
                if h is None:
                    h, base_time, last_frame_time = self.get_base_time()
                    last_capture_frame_time = last_frame_time
                    # put into continuous mode
                    chameleon.trigger(h, True)

                capture_time = time.time()
                if self.camera_settings.depth == 16:
                    im = numpy.zeros((960,1280),dtype='uint16')
                else:
                    im = numpy.zeros((960,1280),dtype='uint8')
                if last_gamma != self.camera_settings.gamma:
                    chameleon.set_gamma(h, self.camera_settings.gamma)
                    last_gamma = self.camera_settings.gamma
                if last_framerate != int(self.camera_settings.framerate):
                    chameleon.set_framerate(h, int(self.camera_settings.framerate))
                    last_framerate = int(self.camera_settings.framerate)

                # capture an image
                frame_time, frame_counter, shutter = chameleon.capture(h, 1000, im)
                if frame_time < last_capture_frame_time:
                    base_time += 128
                last_capture_frame_time = frame_time
                if last_frame_counter != 0:
                    self.frame_loss += frame_counter - (last_frame_counter+1)

                # discard based on process_divider setting
                self.process_counter = (self.process_counter + 1) % self.camera_settings.process_divider
                if self.process_counter % self.camera_settings.process_divider != 0:
                    continue
                
                if self.camera_settings.use_capture_time:
                    img_time = capture_time
                else:
                    img_time = base_time + frame_time

                gammalog.write('%f %f %f %s %u %u\n' % (frame_time,
                                                        frame_time+base_time,
                                                        capture_time,
                                                        cuav_util.frame_time(img_time),
                                                        frame_counter,
                                                        self.camera_settings.gamma))
                gammalog.flush()

                self.save_queue.put((img_time,im))
                self.scan_queue.put((img_time,im))
                self.capture_count += 1
                self.fps = 1.0/(frame_time - last_frame_time)

                if frame_time != last_frame_time:
                    self.framerate = 1.0 / (frame_time - last_frame_time)
                last_frame_time = frame_time
                last_frame_counter = frame_counter
            except chameleon.error, msg:
                self.error_count += 1
                self.error_msg = msg
        if h is not None:
            chameleon.close(h)

    def save_thread(self):
        '''image save thread'''
        raw_dir = os.path.join(self.camera_dir, "raw")
        cuav_util.mkdir_p(raw_dir)
        frame_count = 0
        while not self.unload_event.wait(0.02):
            if self.save_queue.empty():
                continue
            (frame_time,im) = self.save_queue.get()
            rawname = "raw%s" % cuav_util.frame_time(frame_time)
            frame_count += 1
            if self.camera_settings.save_pgm != 0:
                if frame_count % self.camera_settings.save_pgm == 0:
                    chameleon.save_pgm('%s/%s.pgm' % (raw_dir, rawname), im)

    def scan_thread(self):
        '''image scanning thread'''
        while not self.unload_event.wait(0.02):
            try:
                # keep the queue size below 100, so we don't run out of memory
                if self.scan_queue.qsize() > 100:
                    (frame_time,im) = self.scan_queue.get(timeout=0.2)
                (frame_time,im) = self.scan_queue.get(timeout=0.2)
            except Queue.Empty:
                continue

            scan_parms = {}
            for name in self.image_settings.list():
                scan_parms[name] = self.image_settings.get(name)
            scan_parms['SaveIntermediate'] = float(scan_parms['SaveIntermediate'])
            
            t1 = time.time()
            im_full = numpy.zeros((960,1280,3),dtype='uint8')
            im_640 = numpy.zeros((480,640,3),dtype='uint8')
            scanner.debayer(im, im_full)
            scanner.downsample(im_full, im_640)
            if self.camera_settings.fullres:
                img_scan = im_full
            else:
                img_scan = im_640
            regions = scanner.scan(img_scan)
            if self.camera_settings.filter_type=='compactness':
                calculate_compactness = True
            else:
                calculate_compactness = False
            regions = cuav_region.RegionsConvert(regions,
                                                 cuav_util.image_shape(img_scan),
                                                 cuav_util.image_shape(im_full),
                                                 calculate_compactness)
            t2 = time.time()
            self.scan_fps = 1.0 / (t2-t1)
            self.scan_count += 1

            regions = cuav_region.filter_regions(im_full, regions,
                                                 min_score=min(self.camera_settings.minscore,self.camera_settings.minscore2),
                                                 filter_type=self.camera_settings.filter_type)

            self.region_count += len(regions)
            if self.transmit_queue.qsize() < 100:
                self.transmit_queue.put((frame_time, regions, im_full, im_640))

    def get_plane_position(self, frame_time,roll=None):
        '''get a MavPosition object for the planes position if possible'''
        try:
            pos = self.mpos.position(frame_time, 0,roll=roll)
            return pos
        except mav_position.MavInterpolatorException as e:
            print str(e)
            return None

    def log_joe_position(self, pos, frame_time, regions, filename=None, thumb_filename=None):
        '''add to joe.log if possible, returning a list of (lat,lon) tuples
        for the positions of the identified image regions'''
        altitude = self.camera_settings.altitude
        if altitude <= 0:
            altitude = None
        return self.joelog.add_regions(frame_time, regions, pos, filename,
                                       thumb_filename, altitude=altitude)


    def transmit_thread(self):
        '''thread for image transmit to GCS'''
        tx_count = 0
        skip_count = 0
        self.start_aircraft_bsend()

        while not self.unload_event.wait(0.02):
            self.bsend.tick(packet_count=1000, max_queue=self.camera_settings.maxqueue1)
            self.bsend2.tick(packet_count=1000, max_queue=self.camera_settings.maxqueue2)
            self.check_commands(self.bsend)
            self.check_commands(self.bsend2)
            if self.transmit_queue.empty():
                continue

            (frame_time, regions, im_full, im_640) = self.transmit_queue.get()
            if self.camera_settings.roll_stabilised:
                roll=0
            else:
                roll=None
            pos = self.get_plane_position(frame_time, roll=roll)

            # this adds the latlon field to the regions
            self.log_joe_position(pos, frame_time, regions)

            # filter out any regions outside the boundary
            if self.boundary_polygon:
                regions = cuav_region.filter_boundary(regions, self.boundary_polygon, pos)
                regions = cuav_region.filter_regions(im_full, regions, min_score=self.camera_settings.minscore,
                                                     filter_type=self.camera_settings.filter_type)

            self.xmit_queue = self.bsend.sendq_size()
            self.xmit_queue2 = self.bsend2.sendq_size()
            self.efficiency = self.bsend.get_efficiency()
            self.bandwidth_used = self.bsend.get_bandwidth_used()
            self.rtt_estimate = self.bsend.get_rtt_estimate()

            jpeg = None

            if len(regions) > 0:
                lowscore = 0
                highscore = 0
                for r in regions:
                    lowscore = min(lowscore, r.score)
                    highscore = max(highscore, r.score)
                
                if self.camera_settings.transmit:
                    # send a region message with thumbnails to the ground station
                    thumb = None
                    if self.camera_settings.send1:
                        thumb_img = cuav_mosaic.CompositeThumbnail(cv.GetImage(cv.fromarray(im_full)),
                                                                   regions,
                                                                   thumb_size=self.camera_settings.thumbsize)
                        thumb = scanner.jpeg_compress(numpy.ascontiguousarray(cv.GetMat(thumb_img)), self.camera_settings.quality)
                        
                        pkt = ThumbPacket(frame_time, regions, thumb, self.frame_loss, self.xmit_queue, pos)
                        
                        buf = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
                        self.bsend.set_bandwidth(self.camera_settings.bandwidth)
                        self.bsend.set_packet_loss(self.camera_settings.packet_loss)
                        # print("sending thumb len=%u" % len(buf))
                        self.bsend.send(buf,
                                        dest=(self.camera_settings.gcs_address, self.camera_settings.gcs_view_port),
                                        priority=1)

                    # also send thumbnails via 900MHz telemetry
                    if self.camera_settings.send2 and highscore >= self.camera_settings.minscore2:
                        if thumb is None or lowscore < self.camera_settings.minscore2:
                            # remove some of the regions
                            regions = cuav_region.filter_regions(im_full, regions, min_score=self.camera_settings.minscore2,
                                                                 filter_type=self.camera_settings.filter_type)
                            thumb_img = cuav_mosaic.CompositeThumbnail(cv.GetImage(cv.fromarray(im_full)),
                                                                       regions,
                                                                       thumb_size=self.camera_settings.thumbsize)
                            thumb = scanner.jpeg_compress(numpy.ascontiguousarray(cv.GetMat(thumb_img)), self.camera_settings.quality)
                            pkt = ThumbPacket(frame_time, regions, thumb, self.frame_loss, self.xmit_queue, pos)
                            
                            buf = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
                            self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)
                            # print("sending thumb2 len=%u" % len(buf))
                            self.bsend2.send(buf, priority=highscore)

            # Base how many images we send on the send queue size
            send_frequency = self.xmit_queue // 3
            if self.camera_settings.gcs_address is None:
                continue
            if send_frequency != 0 and (tx_count+skip_count) % send_frequency != 0:
                skip_count += 1
                continue
            tx_count += 1
            self.send_image(im_640, frame_time, pos, 0)

    def send_image(self, img, frame_time, pos, priority):
        '''send an image to the GCS'''
        jpeg = scanner.jpeg_compress(img, self.camera_settings.quality)

        # keep filtered image size
        self.jpeg_size = 0.95 * self.jpeg_size + 0.05 * len(jpeg)        

        self.bsend.set_packet_loss(self.camera_settings.packet_loss)
        self.bsend.set_bandwidth(self.camera_settings.bandwidth)
        pkt = ImagePacket(frame_time, jpeg, self.xmit_queue, pos, priority)
        str = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
        # print("sending image len=%u" % len(str))
        self.bsend.send(str,
                        dest=(self.camera_settings.gcs_address, self.camera_settings.gcs_view_port),
                        priority=priority)


    def reload_mosaic(self, mosaic):
        '''reload state into mosaic'''
        regions = []
        last_thumbfile = None
        last_joe = None
        joes = cuav_joe.JoeIterator(self.joelog.filename)
        for joe in joes:
            print joe
            if joe.thumb_filename == last_thumbfile or last_thumbfile is None:
                regions.append(joe.r)
                last_joe = joe
                last_thumbfile = joe.thumb_filename
            else:
                try:
                    composite = cv.LoadImage(last_joe.thumb_filename)
                    thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
                    mosaic.add_regions(regions, thumbs, last_joe.image_filename, last_joe.pos)
                except Exception:
                    pass                
                regions = []
                last_joe = None
                last_thumbfile = None
        if last_joe:
            try:
                composite = cv.LoadImage(last_joe.thumb_filename)
                thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
                mosaic.add_regions(regions, thumbs, last_joe.image_filename, last_joe.pos)
            except Exception:
                pass

    def start_aircraft_bsend(self):
        '''start bsend for aircraft side'''
        if self.bsend is None:
            self.bsend = block_xmit.BlockSender(0, bandwidth=self.camera_settings.bandwidth, debug=False)
        if self.bsend2 is None:
            self.bsocket = MavSocket(self.mpstate.mav_master[0])
            self.bsend2 = block_xmit.BlockSender(mss=96, sock=self.bsocket, dest_ip='mavlink', dest_port=0, backlog=5, debug=False)
            self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)
        

    def start_gcs_bsend(self):
        '''start up block senders for GCS side'''
        if self.bsend is None:
            self.bsend = block_xmit.BlockSender(self.camera_settings.gcs_view_port,
                                                bandwidth=self.camera_settings.bandwidth)
        if self.bsend2 is None:
            self.bsocket = MavSocket(self.mpstate.mav_master[0])
            self.bsend2 = block_xmit.BlockSender(mss=96, sock=self.bsocket, dest_ip='mavlink',
                                                 dest_port=0, backlog=5, debug=False)
            self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)


    def view_thread(self):
        '''image viewing thread - this runs on the ground station'''
        from cuav.lib import cuav_mosaic
        self.start_gcs_bsend()
        view_window = False
        image_count = 0
        thumb_count = 0
        image_total_bytes = 0
        jpeg_total_bytes = 0
        thumb_total_bytes = 0
        region_count = 0
        mosaic = None
        thumbs_received = set()
        view_dir = os.path.join(self.camera_dir, "view")
        thumb_dir = os.path.join(self.camera_dir, "thumb")
        cuav_util.mkdir_p(view_dir)
        cuav_util.mkdir_p(thumb_dir)

        img_window = mp_image.MPImage(title='Camera')

        self.console.set_status('Images', 'Images %u' % image_count, row=6)
        self.console.set_status('Lost', 'Lost %u' % 0, row=6)
        self.console.set_status('Regions', 'Regions %u' % region_count, row=6)
        self.console.set_status('JPGSize', 'JPGSize %.0f' % 0.0, row=6)
        self.console.set_status('XMITQ', 'XMITQ %.0f' % 0.0, row=6)

        self.console.set_status('Thumbs', 'Thumbs %u' % thumb_count, row=7)
        self.console.set_status('ThumbSize', 'ThumbSize %.0f' % 0.0, row=7)
        self.console.set_status('ImageSize', 'ImageSize %.0f' % 0.0, row=7)

        ack_time = time.time()

        self.camera_settings.set_callback(self.camera_settings_callback)
        self.image_settings.set_callback(self.image_settings_callback)

        while not self.unload_event.wait(0.02):
            if not self.viewing:
                if view_window:
                    view_window = False
                continue
        
            tnow = time.time()
            if tnow - ack_time > 0.1:
                self.bsend.tick(packet_count=1000, max_queue=self.camera_settings.maxqueue1)
                self.bsend2.tick(packet_count=1000, max_queue=self.camera_settings.maxqueue2)
                if self.bsend_slave is not None:
                    self.bsend_slave.tick(packet_count=1000)
                    ack_time = tnow
            if not view_window:
                view_window = True
                mosaic = cuav_mosaic.Mosaic(slipmap=self.mpstate.map, C=self.c_params,
                                            camera_settings=self.camera_settings,
                                            image_settings=self.image_settings,
                                            thumb_size=self.camera_settings.mosaic_thumbsize)
                if self.boundary_polygon is not None:
                    mosaic.set_boundary(self.boundary_polygon)
                if self.continue_mode:
                    self.reload_mosaic(mosaic)

            # check for keyboard events
            mosaic.check_events()

            self.check_requested_images(mosaic)

            buf = self.bsend.recv(0)
            if buf is None:
                buf = self.bsend2.recv(0)
                bsend = self.bsend2
                self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)
            else:
                bsend = self.bsend
            if buf is None:
                continue

            try:
                obj = cPickle.loads(str(buf))
                if obj == None:
                    continue
            except Exception as e:
                continue

            if self.camera_settings.gcs_slave is not None:
                if self.bsend_slave is None:
                    self.bsend_slave = block_xmit.BlockSender(0, bandwidth=self.camera_settings.bandwidth*10, debug=False)
                # print("send bsend_slave")
                self.bsend_slave.send(buf,
                                      dest=(self.camera_settings.gcs_slave, self.camera_settings.gcs_view_port),
                                      priority=1)

            if isinstance(obj, ThumbPacket):
                # we've received a set of thumbnails from the plane for a positive hit
                if obj.frame_time in thumbs_received:
                    continue
                thumbs_received.add(obj.frame_time)

                thumb_total_bytes += len(buf)

                # save the thumbnails
                thumb_filename = '%s/v%s.jpg' % (thumb_dir, cuav_util.frame_time(obj.frame_time))
                chameleon.save_file(thumb_filename, obj.thumb)
                composite = cv.LoadImage(thumb_filename)
                thumbs = cuav_mosaic.ExtractThumbs(composite, len(obj.regions))

                # log the joe positions
                filename = '%s/v%s.jpg' % (view_dir, cuav_util.frame_time(obj.frame_time))
                pos = obj.pos
                self.log_joe_position(pos, obj.frame_time, obj.regions, filename, thumb_filename)

                # update the mosaic and map
                mosaic.add_regions(obj.regions, thumbs, filename, pos=pos)

                # update console display
                region_count += len(obj.regions)
                self.frame_loss = obj.frame_loss
                self.xmit_queue = obj.xmit_queue
                thumb_count += 1
            
                self.console.set_status('Lost', 'Lost %u' % self.frame_loss)
                self.console.set_status('Regions', 'Regions %u' % region_count)
                self.console.set_status('XMITQ', 'XMITQ %.0f' % self.xmit_queue)
                self.console.set_status('Thumbs', 'Thumbs %u' % thumb_count)
                self.console.set_status('ThumbSize', 'ThumbSize %.0f' % (thumb_total_bytes/thumb_count))

            if isinstance(obj, ImagePacket):
                # we have an image from the plane
                image_total_bytes += len(buf)

                self.xmit_queue = obj.xmit_queue
                self.console.set_status('XMITQ', 'XMITQ %.0f' % self.xmit_queue)

                # save it to disk
                filename = '%s/v%s.jpg' % (view_dir, cuav_util.frame_time(obj.frame_time))
                chameleon.save_file(filename, obj.jpeg)
                img = cv.LoadImage(filename)
                if img.width == 1280:
                    display_img = cv.CreateImage((640, 480), 8, 3)
                    cv.Resize(img, display_img)
                else:
                    display_img = img

                if obj.pos is not None:
                    mosaic.add_image(obj.frame_time, filename, obj.pos)

                if obj.priority != 0:
                    print("Downloaded image %s (width %u)" % (filename, img.width))
                    if img.width >= 1280:
                        tag_color = (0,0,255)
                    else:
                        tag_color = (0,255,0)
                    mosaic.tag_image(obj.frame_time, tag_color=tag_color)
                    
                cv.ConvertScale(display_img, display_img, scale=self.camera_settings.brightness)
                img_window.set_image(display_img, bgr=True)

                # update console
                image_count += 1
                jpeg_total_bytes += len(obj.jpeg)
                self.jpeg_size = 0.95 * self.jpeg_size + 0.05 * len(obj.jpeg)
                self.console.set_status('Images', 'Images %u' % image_count)
                self.console.set_status('JPGSize', 'JPG Size %.0f' % (jpeg_total_bytes/image_count))
                self.console.set_status('ImageSize', 'ImageSize %.0f' % (image_total_bytes/image_count))

            if isinstance(obj, CommandPacket):
                self.handle_command_packet(obj, bsend)

            if isinstance(obj, CommandResponse):
                print('REMOTE: %s' % obj.response)

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
        if self.capture_thread_h is not None:
            self.capture_thread.join(1.0)
            self.save_thread.join(1.0)
            self.scan_thread1.join(1.0)
            self.scan_thread2.join(1.0)
            self.transmit_thread.join(1.0)
        if self.view_thread_h is not None:
            self.view_thread.join(1.0)
        print('camera unload OK')

    def handle_command_packet(self, obj, bsend):
        '''handle CommandPacket from other end'''
        stdout_saved = sys.stdout
        buf = cStringIO.StringIO()
        sys.stdout = buf
        self.mpstate.functions.process_stdin(obj.command, immediate=True)
        sys.stdout = stdout_saved
        pkt = CommandResponse(str(buf.getvalue()))
        buf = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
        bsend.send(buf, priority=10000)

    def check_commands(self, bsend):
        '''check for remote commands'''
        if bsend is None:
            return
        buf = bsend.recv(0)
        if buf is None:
            return
        try:
            obj = cPickle.loads(str(buf))
            if obj == None:
                return
        except Exception as e:
            return

        if isinstance(obj, CommandPacket):
            self.handle_command_packet(obj, bsend)

        if isinstance(obj, CommandResponse):
            print('REMOTE: %s' % obj.response)

        if isinstance(obj, ImageRequest):
            self.handle_image_request(obj, bsend)            

        if isinstance(obj, ChangeCameraSetting):
            self.camera_settings.set(obj.name, obj.value)

        if isinstance(obj, ChangeImageSetting):
            self.image_settings.set(obj.name, obj.value)

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        if self.mpstate.status.watch in ["camera","queue"] and time.time() > self.last_watch+1:
            self.last_watch = time.time()
            self.cmd_camera(["status" if self.mpstate.status.watch == "camera" else "queue"])
        # update position interpolator
        self.mpos.add_msg(m)
        if m.get_type() in [ 'DATA16', 'DATA32', 'DATA64', 'DATA96' ]:
            if self.bsocket is not None:
                self.bsocket.incoming.append(m)
        if m.get_type() == 'SYSTEM_TIME' and self.camera_settings.clock_sync and self.capture_thread_h is not None:
            # optionally sync system clock on the capture side
            self.sync_gps_clock(m.time_unix_usec)

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

    def check_requested_images(self, mosaic):
        '''check if the user has requested download of an image'''
        requests = mosaic.get_image_requests()
        for frame_time in requests.keys():
            fullres = requests[frame_time]
            pkt = ImageRequest(frame_time, fullres)
            buf = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
            self.start_gcs_bsend()
            print("Requesting image %s" % frame_time)
            if self.camera_settings.use_bsend2:
                if self.bsend2 is None:
                    print("bsend2 not initialised")
                    return
                self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)
                self.bsend2.send(buf, priority=10000)
            else:
                if self.bsend is None:
                    print("bsend not initialised")
                    return
                self.bsend.send(buf, priority=10000)

    def handle_image_request(self, obj, bsend):
        '''handle ImageRequest from GCS'''
        rawname = "raw%s" % cuav_util.frame_time(obj.frame_time)
        raw_dir = os.path.join(self.camera_dir, "raw")
        filename = '%s/%s.pgm' % (raw_dir, rawname)
        if not os.path.exists(filename):
            print("No file: %s" % filename)
            return
        try:
            img = cuav_util.LoadImage(filename)
            img = numpy.asarray(cv.GetMat(img))
        except Exception:
            return
        if not obj.fullres:
            im_640 = numpy.zeros((480,640,3),dtype='uint8')
            scanner.downsample(img, im_640)
            img = im_640
        print("Sending image %s" % filename)
        self.send_image(img, obj.frame_time, None, 10000)

    def send_packet(self, pkt):
        '''send a packet from GCS'''
        buf = cPickle.dumps(pkt, cPickle.HIGHEST_PROTOCOL)
        self.start_gcs_bsend()
        if self.camera_settings.use_bsend2:
            if self.bsend2 is None:
                print("bsend2 not initialised")
                return
            self.bsend2.set_bandwidth(self.camera_settings.bandwidth2)
            self.bsend2.send(buf, priority=10000)
        else:
            if self.bsend is None:
                print("bsend not initialised")
                return
            self.bsend.send(buf, priority=10000)

    def camera_settings_callback(self, setting):
        '''called on a changed camera setting'''
        pkt = ChangeCameraSetting(setting.name, setting.value)
        self.send_packet(pkt)

    def image_settings_callback(self, setting):
        '''called on a changed image setting'''
        pkt = ChangeImageSetting(setting.name, setting.value)
        self.send_packet(pkt)

def init(mpstate):
    '''initialise module'''
    return CameraModule(mpstate)
