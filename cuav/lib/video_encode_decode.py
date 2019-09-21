#!/usr/bin/env python
'''
encode and then decode video, displaying original and decoded frames
'''

import video_encode
import video_play
import io
import argparse
import sys
import cv2
import time
from MAVProxy.modules.lib import mp_image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--delay", type=float, default=1.0)
ap.add_argument("--crop", type=str, default=None)
ap.add_argument("imgs", type=str, nargs='+')
args = ap.parse_args()

if len(args.imgs) < 2:
    print("Need at least 2 images")
    sys.exit(1)
    
encoder = video_encode.VideoWriter()
decoder = video_play.VideoReader()

if args.crop:
    encoder.set_cropstr(args.crop)

viewer_in = mp_image.MPImage(title='Origin', width=200, height=200, auto_size=True)
viewer_out = mp_image.MPImage(title='Decoded', width=200, height=200, auto_size=True)

idx = 0
timestamp_ms = 0

while True:
    time.sleep(args.delay)
    img = cv2.imread(args.imgs[idx])
    cropped = encoder.crop_image(img)

    viewer_in.set_image(cropped)
    enc = encoder.add_image(img, timestamp_ms)

    timestamp_ms += 1000
    idx = (idx + 1) % len(args.imgs)

    encoder.report()

    s = io.BytesIO()
    s.write(enc)
    s.seek(0)
    (img,dt) = decoder.get_image(s)
    if img is None:
        continue
    viewer_out.set_image(img)

