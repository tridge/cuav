#!/usr/bin/env python

'''
play back a mavlink log and set of images as a
realtime mavlink stream

Useful for testing the ground station using previously logged data and images

Due to Windows lacking symlinking (and funky timestamping for mavlink messages), this script 
isn't compatible with Windows
'''

import sys, time, os, struct, glob

from cuav.lib import cuav_util
from argparse import ArgumentParser
from pymavlink import mavutil


class ImageFile:
    def __init__(self, frame_time, filename):
        self.frame_time = frame_time
        self.filename = filename

def scan_image_directory(dirname):
    '''scan a image directory, extracting frame_time and filename
    as a list of tuples'''
    ret = []
    types = ('*.png', '*.jpeg', '*.jpg')
    for tp in types:
        for f in glob.iglob(os.path.join(dirname, tp)):
            ret.append(ImageFile(cuav_util.parse_frame_time(f), f))
    ret.sort(key=lambda f: f.frame_time)
    return ret

def playback(filename, images, argout, argbaudrate, argcondition, argspeedup, linkname="capture.jpg"):
    '''playback one file'''
    mlog = mavutil.mavlink_connection(filename, robust_parsing=True)
    mout = mavutil.mavlink_connection(argout, input=False, baud=argbaudrate)

    # get first message
    msg = mlog.recv_match(condition=argcondition)
    last_timestamp = msg._timestamp
    last_print = time.time()

    # skip any older images
    while len(images) and images[0].frame_time < msg._timestamp:
        images.pop(0)

    params = []
    param_send = []

    while True:
        msg = mlog.recv_match(condition=argcondition)
        if msg is None:
            return
        if msg.get_type().startswith('DATA'):
            continue
        if msg.get_type() == 'PARAM_VALUE':
            params.append(msg)
        mout.write(msg.get_msgbuf())
        deltat = msg._timestamp - last_timestamp
        if len(images) == 0 or images[0].frame_time > msg._timestamp + 2:
            # run at high speed except for the portions where we have images
            deltat /= 60
        time.sleep(max(min(deltat/argspeedup, 5), 0))
        last_timestamp = msg._timestamp
        if time.time() - last_print > 2.0:
            print('%s' % (time.asctime(time.localtime(msg._timestamp))))
            last_print = time.time()

        if len(images) and msg._timestamp > images[0].frame_time:
            img = images.pop(0)
            try:
                os.unlink(linkname)
            except Exception:
                pass
            os.symlink(img.filename, linkname)
            print(img.filename)

        # check for parameter fetch messages
        msg = mout.recv_msg()
        if msg and msg.get_type() == 'PARAM_REQUEST_LIST':
            print("Sending %u parameters" % len(params))
            param_send = params[:]

        if len(param_send) != 0:
            p = param_send.pop(0)
            mout.write(p.get_msgbuf())

if __name__ == '__main__':
    parser = ArgumentParser(description="play back a mavlink log and set of images as a mavlink stream")
    parser.add_argument("imagedir", default=None, help='image directory')
    parser.add_argument("logdir", default=None, help='log file')
    parser.add_argument("--out",   help="MAVLink output port (IP:port)", default='udpout:127.0.0.1:14550')
    parser.add_argument("--baudrate", type=int, default=57600, help='baud rate')
    parser.add_argument("--condition", default=None, help='condition on mavlink log')
    parser.add_argument("--speedup", type=float, default=1.0, help='playback speedup')
    parser.add_argument("--loop", action='store_true', default=False, help='playback in a loop')
    
    args = parser.parse_args()
    
    #Check if we're running under Windows:
    if sys.platform.startswith('win'):
        print("This script is not compatible with Windows")
        sys.exit()
    
    while True:
        images = scan_image_directory(args.imagedir)
        if len(images) == 0:
            print("No images supplied")
            sys.exit(0)
        print("Found %u images for %.1f minutes" % (len(images),
                                                    (images[-1].frame_time-images[0].frame_time)/60.0))
        playback(args.logdir, images, args.out, args.baudrate, args.condition, args.speedup)
        if not args.loop:
            break
