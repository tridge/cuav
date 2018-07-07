#!/usr/bin/env python
'''realtime imaging control via MAVProxy, air side
It takes in sent images from camera_air and displays them
via a GUI'''


import time, threading, os, cPickle
import functools, cv2

from MAVProxy.modules.lib import mp_module, mp_image
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting
from MAVProxy.modules.mavproxy_map import mp_slipmap

from cuav.lib import cuav_mosaic, cuav_util, cuav_joe, block_xmit, cuav_command
from cuav.camera.cam_params import CameraParams


class CameraGroundModule(mp_module.MPModule):
    def __init__(self, mpstate):
        super(CameraGroundModule, self).__init__(mpstate,
                                                 "camera_ground",
                                                 "cuav camera control (ground)",
                                                 public=True)

        self.unload_event = threading.Event()
        self.unload_event.clear()

        self.view_thread = None
        self.handled_timestamps = {}

        self.camera_settings = MPSettings(
            [MPSetting('air_address',
                       str,
                       "",
                       'Air Addresses in RemIP:RemPort:LocalPort:Bandwidth\
                       format (127.0.0.1:1440:1234:45, ...)',
                       tab='GCS'),
             MPSetting('brightness', float, 1.0, 'Display Brightness',
                       range=(0.1, 10), increment=0.1,
                       digits=2, tab='Display'),
             MPSetting('debug', bool, False, 'debug enable'),
             MPSetting('camparms', str, None, 'camera parameters file (json)'),
             MPSetting('mosaic_thumbsize', int, 35, 'Mosaic Thumbnail Size',
                       range=(10, 200), increment=1),
             MPSetting('maxqueue', int, 100, 'Maximum images queue'),
             MPSetting('target_latitude', float, 0, 'filter detected images to latitude', tab='Filter to Location'),
             MPSetting('target_longitude', float, 0, 'filter detected images to longitude', tab='Filter to Location'),
             MPSetting('target_radius', float, 0, 'filter detected images to radius', tab='Filter to Location'),
            ],
            title='Camera (ground) Settings'
        )

        self.viewing = False

        self.boundary = None
        self.boundary_polygon = None

        #just make a temp dir for the downloaded images and thumbs
        #this folder is deleted when the module is unloaded
        #self.camera_dir = tempfile.mkdtemp()
        self.camera_dir = self.mpstate.status.logdir

        self.bsend = []
        #self.last_minscore = None
        self.mosaic = None
        self.last_heartbeat = time.time()

        self.joelog = None

        self.c_params = None

        self.add_command('camera', self.cmd_camera,
                         'camera control',
                         ['<status|view|boundary|remoteset>',
                          'set (CAMERASETTING)'])
        self.add_completion_function('(CAMERASETTING)', self.settings.completion)
        self.add_completion_function('(CAMERASETTING)', self.camera_settings.completion)
        print("camera (ground) initialised")

    def cmd_camera(self, args):
        '''camera commands'''
        usage = "usage: camera <status|view|boundary|set|remoteset>"
        if len(args) == 0:
            print(usage)
            return
        elif args[0] == "status":
            print("Cap imgs: regions:%u" % (self.region_count))
        elif args[0] == "view":
            #check cam params
            if not os.path.isabs(self.camera_settings.camparms):
                print("Error - camera params must use absolute path")
                return
            if not self.check_camera_parms():
                print("Error - incorrect camera params")
                return
            if self.mpstate.map is None:
                print("Error - load map module first")
                return
            if not self.viewing:
                print("Starting image viewer")
            self.joelog = cuav_joe.JoeLog(os.path.join(self.camera_dir,
                                                       'joe_ground.log'),
                                          append=self.continue_mode)
            if self.view_thread is None:
                self.view_thread = self.start_thread(self.view_threadfunc)
            self.viewing = True
        elif args[0] == "set":
            self.camera_settings.command(args[1:])
        elif args[0] == "boundary":
            if len(args) != 2:
                print("boundary=%s" % self.boundary)
            else:
                self.boundary = args[1]
                self.boundary_polygon = cuav_util.polygon_load(self.boundary)
                if self.mpstate.map is not None:
                    self.mpstate.map.add_object(mp_slipmap.SlipPolygon('boundary',
                                                                       self.boundary_polygon,
                                                                       layer=1, linewidth=2,
                                                                       colour=(0, 0, 255)))
        elif args[0] == "remoteset":
            if len(args) != 3:
                print("Error in command")
                return
            if self.bsend == []:
                print("Error - no active connection to cuav_air")
                return
            #send remote command
            pkt = cuav_command.ChangeCameraSetting(args[1], args[2])
            self.send_packet(pkt)


    def transmit_thread(self):
        '''thread for send/recieve to air side'''
        pass

    def check_camera_parms(self):
        '''check for change in camera parameters'''
        if self.camera_settings.camparms is None:
            return False
        if os.path.isfile(self.camera_settings.camparms):
            try:
                self.c_params = CameraParams.fromfile(self.camera_settings.camparms)
                return True
            except:
                return False
        else:
            return False

    def reload_mosaic(self, mosaic):
        '''reload state into mosaic'''
        regions = []
        last_thumbfile = None
        last_joe = None
        joes = []
        if os.path.isfile(self.joelog.filename):
            joes = cuav_joe.JoeIterator(self.joelog.filename)
        for joe in joes:
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

    def start_gcs_bsend(self):
        '''start up block senders for GCS side'''
        if len(self.bsend) == 0:
            for lnk in self.camera_settings.air_address.split(','):
                try:
                    [remoteip, remoteport, localport, bw] = lnk.split(':')
                    newbsnd = block_xmit.BlockSender(bandwidth=int(bw), debug=False,
                                                     dest_ip=remoteip,
                                                     dest_port=int(remoteport),
                                                     port=int(localport))
                    self.bsend.append(newbsnd)
                except:
                    print("Bad Air endpoint (must be remIP:remport:localport:bw): " + str(lnk))
                    pass

        # send an initial packet to open the link
        self.send_packet(cuav_command.CommandPacket(''))
        self.send_heartbeat()

    def view_threadfunc(self):
        '''image viewing thread - this runs on the ground station'''
        self.start_gcs_bsend()
        self.image_count = 0
        self.thumb_count = 0
        self.image_total_bytes = 0
        #jpeg_total_bytes = 0
        self.thumb_total_bytes = 0
        self.region_count = 0
        self.mosaic = None
        self.thumbs_received = set()
        # the downloaded thumbs and views are stored in a temp dir
        self.view_dir = os.path.join(self.camera_dir, "view")
        #self.thumb_dir = os.path.join(self.camera_dir, "thumb")
        cuav_util.mkdir_p(self.view_dir)
        #cuav_util.mkdir_p(self.thumb_dir)

        self.console.set_status('Images', 'Images %u' % self.image_count, row=6)
        self.console.set_status('Regions', 'Regions %u' % self.region_count, row=6)
        self.console.set_status('JPGSize', 'JPGSize %.0f' % 0.0, row=6)

        self.console.set_status('Thumbs', 'Thumbs %u' % self.thumb_count, row=7)
        self.console.set_status('ThumbSize', 'ThumbSize %.0f' % 0.0, row=7)
        self.console.set_status('ImageSize', 'ImageSize %.0f' % 0.0, row=7)

        self.mosaic = cuav_mosaic.Mosaic(slipmap=self.mpstate.map, C=self.c_params,
                                         camera_settings=None,
                                         image_settings=None,
                                         thumb_size=self.camera_settings.mosaic_thumbsize)

        while not self.unload_event.wait(0.02):

            if self.boundary_polygon is not None:
                self.mosaic.set_boundary(self.boundary_polygon)
            if self.continue_mode:
                self.reload_mosaic(self.mosaic)

            # check for keyboard events
            self.mosaic.check_events()

            self.check_requested_images(self.mosaic)
            #check for any new packets
            for bsnd in self.bsend:
                bsnd.tick(packet_count=1000, max_queue=self.camera_settings.maxqueue)
                self.check_commands(bsnd)
            self.send_heartbeats()

        #ensure the mosiac is closed at end of thread
        if self.mosaic.image_mosaic:
            self.mosaic.image_mosaic.terminate()


    def send_heartbeats(self):
        '''possibly send heartbeat msgs'''
        now = time.time()
        if now - self.last_heartbeat > 5:
            self.last_heartbeat = now
            self.send_heartbeat()


    def check_commands(self, bsend):
        '''check for remote commands'''
        if bsend is None:
            return
        buf = bsend.recv(0)
        if buf is None:
            return
        try:
            obj = cPickle.loads(str(buf))
            if obj is None:
                return
        except Exception as e:
            return

        if isinstance(obj, cuav_command.StampedCommand):
            if obj.timestamp in self.handled_timestamps:
                # we've seen this packet before, discard
                return
            self.handled_timestamps[obj.timestamp] = time.time()

        if isinstance(obj, cuav_command.ThumbPacket):
            # we've received a set of thumbnails from the plane for a positive hit
            if obj.frame_time not in self.thumbs_received:
                self.thumbs_received.add(obj.frame_time)

            self.thumb_total_bytes += len(buf)

            # add the thumbnails to the mosaic
            thumbdec = cv2.imdecode(obj.thumb, 1)
            if thumbdec is None:
                pass
            thumbs = cuav_mosaic.ExtractThumbs(thumbdec, len(obj.regions))

            # log the joe positions
            # note the filename is where the fullsize image would be downloaded
            # to, if requested
            filename = os.path.join(self.view_dir, cuav_util.frame_time(obj.frame_time)) + ".jpg"
            self.log_joe_position(obj.pos, obj.frame_time, obj.regions, filename, None)

            # update the mosaic and map
            self.mosaic.add_regions(obj.regions, thumbs, filename, obj.pos)

            # update console display
            self.region_count += len(obj.regions)
            self.thumb_count += 1

            self.console.set_status('Regions', 'Regions %u' % self.region_count, row=6)
            self.console.set_status('Thumbs', 'Thumbs %u' % self.thumb_count, row=7)
            self.console.set_status('ThumbSize', 'ThumbSize %.0f' %
                                    (self.thumb_total_bytes/self.thumb_count), row=7)

        if isinstance(obj, cuav_command.ImagePacket):
            # we have an image from the plane
            self.image_total_bytes += len(buf)

            #save to file
            imagedec = cv2.imdecode(obj.jpeg, 1)
            ff = os.path.join(self.view_dir, cuav_util.frame_time(obj.frame_time)) + ".jpg"
            write_param = [int(cv2.IMWRITE_JPEG_QUALITY), 99]
            cv2.imwrite(ff, imagedec, write_param)
            self.mosaic.tag_image(obj.frame_time)

            # update console
            self.image_count += 1
            color = 'black'
            self.console.set_status('Images', 'Images %u' % self.image_count, row=6, fg=color)
            self.console.set_status('ImageSize', 'ImageSize %.0f' %
                                    (self.image_total_bytes/self.image_count), row=7)

        if isinstance(obj, cuav_command.CommandPacket):
            pass

        if isinstance(obj, cuav_command.CommandResponse):
            print('REMOTE: %s' % obj.response)

        if isinstance(obj, cuav_command.CameraMessage):
            self.say(obj.msg)

    def log_joe_position(self, pos, frame_time, regions, filename=None, thumb_filename=None):
        '''add to joe_ground.log if possible, returning a list of (lat,lon) tuples
        for the positions of the identified image regions'''
        return self.joelog.add_regions(frame_time, regions, pos, filename,
                                       thumb_filename, altitude=None, C=self.c_params)

    def start_thread(self, fn):
        '''start a thread running'''
        t = threading.Thread(target=fn)
        t.daemon = True
        t.start()
        return t

    def unload(self):
        '''unload module'''
        self.unload_event.set()
        if self.view_thread is not None:
            self.view_thread.join(1.0)
        #shutil.rmtree(self.camera_dir)
        print('camera unload OK')

    def mavlink_packet(self, m):
        '''handle an incoming mavlink packet'''
        pass

    def check_requested_images(self, mosaic):
        '''check if the user has requested download of an image'''
        requests = mosaic.get_image_requests()
        for frame_time in requests.keys():
            fullres = requests[frame_time]
            pkt = cuav_command.ImageRequest(frame_time, fullres)
            print("Requesting image %s" % frame_time)
            self.send_object(pkt, priority=10000)

    def send_packet(self, pkt):
        '''send a packet from GCS'''
        self.send_object(pkt, priority=10000)

    def send_heartbeat(self):
        '''send a heartbeat'''
        pkt = cuav_command.HeartBeat()
        self.send_packet(pkt)

    def send_message(self, msg):
        '''send a message'''
        pkt = cuav_command.CameraMessage(msg)
        self.send_packet(pkt)

    def send_object_complete(self, obj):
        '''called on complete of an send_object, cancelling send on other link'''
        if obj.blockid is not None:
            for bsnd in self.bsend:
                bsnd.cancel(obj.blockid)

    def send_object(self, obj, priority):
        buf = cPickle.dumps(obj, cPickle.HIGHEST_PROTOCOL)
        #only send if the queue is not clogged
        for bsnd in self.bsend:
            if bsnd.sendq_size() < self.camera_settings.maxqueue:
                obj.blockid = bsnd.send(buf, priority=priority,
                                        callback=functools.partial(self.send_object_complete, obj))



def init(mpstate):
    '''initialise module'''
    return CameraGroundModule(mpstate)
