#!/usr/bin/env python

'''
an image playback script that creates images from target thumbnails and base images
'''

import sys, time, os, struct, glob, pickle, random, cv2

from cuav.lib import cuav_util, mav_position
from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
from pymavlink import mavutil
from cuav.camera import cam_params

from math import *

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

def find_xy_in_image(lat, lon, pos, Xres, Yres, C_params):
    '''find (x,y) coordinates in image that match a position. Very simple method'''
    # find best X
    bestX = -1
    bestDist = -1
    for x in range(0, Xres, 10):
        latlon = cuav_util.gps_position_from_xy(x, Yres/2, pos, C_params, pos.altitude)
        if latlon is None:
            return None
        (lat1,lon1) = latlon
        dist = cuav_util.gps_distance(lat, lon, lat1, lon1)
        if bestDist < 0 or dist < bestDist:
            bestX = x
            bestDist = dist
    if bestX <= 20 or bestX >= Xres-20:
        return None
    bestY = -1
    bestDist = -1
    for y in range(0, Yres, 10):
        latlon = cuav_util.gps_position_from_xy(bestX, y, pos, C_params, pos.altitude)
        if latlon is None:
            return None
        (lat1,lon1) = latlon
        dist = cuav_util.gps_distance(lat, lon, lat1, lon1)
        if bestDist < 0 or dist < bestDist:
            bestY = y
            bestDist = dist
    if bestY <= 20 or bestY >= Yres-20:
        return None
    return (bestX, bestY)

def playback(mavcon, images, targets, target_lat, target_lon, C_params):
    '''create synthetic images matching target position'''
    mlog = mavutil.mavlink_connection(mavcon)

    # get first message
    msg = mlog.recv_match(type='ATTITUDE', blocking=True)
    mavtime = msg.time_boot_ms * 0.001
    last_print = time.time()
    last_image = mavtime

    Xres = 1640
    Yres = 1232

    try:
        os.mkdir("tmpimages")
    except Exception:
        pass

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
                pos = mav_position.MavPosition(lat, lon, alt, roll, pitch, yaw, time.time())
                
                imgfile = random.choice(images)
                img = cv2.imread(imgfile)
                filename = os.path.join("tmpimages", cuav_util.frame_time(time.time()) + ".jpg")
                
                # see if the target should be in the image
                xy = find_xy_in_image(target_lat, target_lon, pos, Xres, Yres, C_params)
                if xy is None:
                    # when we are not modifying the image use a hardlink to save space
                    os.link(imgfile, filename)
                else:
                    # overlay a random target on the image
                    (x,y) = xy
                    tgtfile = random.choice(targets)
                    tgt = cv2.imread(tgtfile)
                    img[y:y+tgt.shape[0], x:x+tgt.shape[1]] = tgt
                    cv2.imwrite(filename, img)
                # create symlink
                try:
                    os.unlink("capture.jpg")
                except Exception:
                    pass
                os.symlink(filename, "capture.jpg")
                print(filename, xy)

if __name__ == '__main__':
    parser = ArgumentParser(description="play back a mavlink log and set of images as a mavlink stream")
    parser.add_argument("--imagedir", default="cmac_images", help='image directory')
    parser.add_argument("--targetdir", default="targets", help='target directory')
    parser.add_argument("--target-lat", type=float, default=-35.362846, help='target latitude')
    parser.add_argument("--target-lon", type=float, default=149.164272, help='target longitude')
    parser.add_argument("--mavcon",   help="MAVLink input port (IP:port)", default='tcp:127.0.0.1:5763')
    parser.add_argument("--camera-params", default='../../../cuav/data/PiCamV2/params_half.json', type=file, help="camera calibration json file from OpenCV")
    
    args = parser.parse_args()

    images = glob.glob(os.path.join(args.imagedir, "*.jpg"))
    targets = glob.glob(os.path.join(args.targetdir, "*.png"))

    C_params = cam_params.CameraParams.fromfile(args.camera_params.name)

    print("Found %u images and %u targets" % (len(images), len(targets)))
    playback(args.mavcon, images, targets, args.target_lat, args.target_lon, C_params)
