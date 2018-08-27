#!/usr/bin/env python

'''an image playback script that uses mavpos.dat to play images that match the
location and attitude of an aircraft in SITL
'''

import sys, time, os, struct, glob, pickle

from cuav.lib import cuav_util
from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
from pymavlink import mavutil

from math import *

class ImageFile:
    def __init__(self, frame_time, filename, pos):
        self.frame_time = frame_time
        self.filename = filename
        self.pos = pos

def scan_image_directory(dirname, mavpos):
    '''scan a image directory, extracting frame_time and filename
    as a list of tuples'''
    ret = []
    types = ('*.png', '*.jpeg', '*.jpg')
    for tp in types:
        for f in glob.iglob(os.path.join(dirname, tp)):
            bname = os.path.basename(f)
            if bname in mavpos:
                ret.append(ImageFile(cuav_util.parse_frame_time(f), f, mavpos[bname]))
    ret.sort(key=lambda f: f.frame_time)
    return ret

def find_best_image(images, lat, lon, alt, roll, pitch, yaw):
    dist_margin = 20.0
    alt_margin = 10.0
    roll_margin = 10.0
    pitch_margin = 10.0
    yaw_margin = 10.0

    for img in images:
        pos = img.pos
        dist = mp_util.gps_distance(lat, lon, pos.lat, pos.lon)
        if dist > dist_margin:
            continue
        if abs(alt - pos.altitude) > alt_margin:
            continue
        if abs(roll - pos.roll) > roll_margin:
            continue
        if abs(pitch - pos.pitch) > pitch_margin:
            continue
        if abs(yaw - pos.yaw) > yaw_margin:
            continue
        return img
    return None
        

def playback(images, mavcon):
    '''playback images matching position and attitude'''
    mlog = mavutil.mavlink_connection(mavcon)

    # get first message
    msg = mlog.recv_match(type='ATTITUDE', blocking=True)
    mavtime = msg.time_boot_ms * 0.001
    last_print = time.time()
    last_image = mavtime

    # find min and max latitude
    min_lat = -90.0
    max_lat = 90.0
    for img in images:
        min_lat = min(min_lat, img.pos.lat)
        max_lat = max(max_lat, img.pos.lat)

    # bucket the images by latitude
    num_buckets = 100
    buckets = {}
    for i in range(num_buckets):
        for img in images:
            b = int(num_buckets * (img.pos.lat - min_lat) / (max_lat - min_lat))
            if not b in buckets:
                buckets[b] = []
            buckets[b].append(img)
    
    while True:
        msg = mlog.recv_match(blocking=True)
        if not msg:
            break
        mtype = msg.get_type()
        if mtype == "ATTITUDE":
            mavtime = msg.time_boot_ms*0.001
            if mavtime - last_image > 0.8:
                if not 'GLOBAL_POSITION_INT' in mlog.messages:
                    continue
                global_position_int = mlog.messages['GLOBAL_POSITION_INT']
                attitude = mlog.messages['ATTITUDE']
                lat = global_position_int.lat*1.0e-7
                lon = global_position_int.lon*1.0e-7
                alt = global_position_int.relative_alt*0.001
                roll = degrees(attitude.roll)
                pitch = degrees(attitude.pitch)
                yaw = degrees(attitude.yaw)
                if lat > max_lat or lat < min_lat:
                    continue
                b = int(num_buckets * (lat - min_lat) / (max_lat - min_lat))
                img = find_best_image(buckets[b], lat, lon, alt, roll, pitch, yaw)
                if img is not None:
                    try:
                        os.unlink("capture.jpg")
                    except Exception:
                        pass
                    os.symlink(img.filename, "capture.jpg")
                    print(img.filename)
                    last_image = mavtime
        

if __name__ == '__main__':
    parser = ArgumentParser(description="play back a mavlink log and set of images as a mavlink stream")
    parser.add_argument("imagedir", default=None, help='image directory')
    parser.add_argument("--mavcon",   help="MAVLink input port (IP:port)", default='tcp:127.0.0.1:5763')
    
    args = parser.parse_args()
    imagedir = args.imagedir
    
    #Check if we're running under Windows:
    if sys.platform.startswith('win'):
        print("This script is not compatible with Windows")
        sys.exit()

    mavposfile = os.path.join(imagedir, "mavpos.dat")
    if not os.path.exists(mavposfile):
        print("%s not found" % mavposfile)
        sys.exit()
        
    mavpos = pickle.loads(open(mavposfile,"r").read())
    
    images = scan_image_directory(args.imagedir, mavpos)
    if len(images) == 0:
        print("No images supplied")
        sys.exit(0)
        
    print("Found %u images for %.1f minutes" % (len(images),
                                            (images[-1].frame_time-images[0].frame_time)/60.0))
    playback(images, args.mavcon)
