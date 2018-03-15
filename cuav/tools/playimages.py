#!/usr/bin/env python

'''
read a mavlink log and set of images, then present the best
matching images for a position on a mavlink stream
'''

import sys, time, os, struct, glob, math

from cuav.lib import cuav_util

from optparse import OptionParser
parser = OptionParser("playimages.py [options] <flight-log>")

parser.add_option("--master",   help="MAVLink input port (IP:port)", default='127.0.0.1:14550')
parser.add_option("--baudrate", type='int', default=57600, help='baud rate')
parser.add_option("--imagedir", default=None, help='raw image directory')
(opts, args) = parser.parse_args()

from pymavlink import mavutil

if len(args) < 1:
    parser.print_help()
    sys.exit(1)

class ImageFile:
    def __init__(self, frame_time, filename):
        self.frame_time = frame_time
        self.filename = filename
        self.pos = None

def scan_image_directory(dirname):
    '''scan a image directory, extracting frame_time and filename
    as a list of tuples'''
    ret = []
    pattern = '*.pgm'
    for f in glob.iglob(os.path.join(dirname, pattern)):
        ret.append(ImageFile(cuav_util.parse_frame_time(f), f))
    ret.sort(key=lambda f: f.frame_time)
    return ret

class PosMap:
    '''an object to map from position to an image'''
    def __init__(self, images):
        self.images = images
        self.latset = {}
        self.lonset = {}

    def scan(self, filename):
        '''scan a tlog file for positions'''
        mlog = mavutil.mavlink_connection(filename, robust_parsing=True)
        idx = 0
        print("Scanning %u images" % len(self.images))
        while idx < len(self.images):
            msg = mlog.recv_match(type='GLOBAL_POSITION_INT')
            if msg is None:
                break
            bearing = msg.hdg*0.01
            speed = math.sqrt(msg.vx**2 + msg.vy**2)
            while idx < len(self.images) and msg._timestamp > self.images[idx].frame_time:
                dt = msg._timestamp - self.images[idx].frame_time
                pos = (msg.lat*1.0e-7, msg.lon*1.0e-7)
                pos = cuav_util.gps_newpos(pos[0], pos[1], bearing, speed*dt)
                self.images[idx].pos = pos
                latint = int(pos[0]*1000)
                lonint = int(pos[1]*1000)
                if not latint in self.latset:
                    self.latset[latint] = set()
                if not lonint in self.lonset:
                    self.lonset[lonint] = set()
                self.latset[latint].add(idx)
                self.lonset[lonint].add(idx)
                idx += 1
        print("Scanned %u images" % idx)

    def best_image(self, pos):
        '''return the best image for a position'''
        latint = int(pos[0]*1000)
        lonint = int(pos[1]*1000)
        if not latint in self.latset or not lonint in self.lonset:
            return None
        s = self.latset[latint].intersection(self.lonset[lonint])
        if len(s) == 0:
            return None
        bestdist = None
        bestidx = None
        for idx in s:
            ipos = self.images[idx].pos
            if ipos is None:
                continue
            dist = cuav_util.gps_distance(pos[0], pos[1], ipos[0], ipos[1])
            if bestdist is None or dist < bestdist:
                bestdist = dist
                bestidx = idx
        if bestdist is None:
            return None
        print(bestdist)
        return self.images[bestidx]

def playimages(device, pmap):
    '''play images for a connection'''
    print("Opening %s" % device)
    min = mavutil.mavlink_connection(device, input=True, baud=opts.baudrate)

    while True:
        msg = min.recv_match(type='GLOBAL_POSITION_INT')
        if msg is None:
            time.sleep(0.1)
            continue
        pos = (msg.lat*1.0e-7, msg.lon*1.0e-7)
        img = pmap.best_image(pos)
        if img is None:
            continue

        try:
            os.unlink('fake_chameleon.tmp')
        except Exception:
            pass
        os.symlink(img.filename, 'fake_chameleon.tmp')
        os.rename('fake_chameleon.tmp', 'fake_chameleon.pgm')
        print(img.filename)

images = scan_image_directory(opts.imagedir)
pmap = PosMap(images)
pmap.scan(args[0])
playimages(opts.master, pmap)
