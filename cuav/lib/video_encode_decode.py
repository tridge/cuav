#!/usr/bin/env python
'''
encode and then decode video, displaying original and decoded frames
'''

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
ap.add_argument("--x265", action='store_true', default=False)
ap.add_argument("imgs", type=str, nargs='+')
args = ap.parse_args()

if len(args.imgs) < 2:
    print("Need at least 2 images")
    sys.exit(1)

if args.x265:
    import video_encode_x265
    import video_decode_dec265

    encoder = video_encode_x265.VideoWriter()
    decoder = video_decode_dec265.VideoReader()
else:
    import video_encode
    import video_play

    encoder = video_encode.VideoWriter()
    decoder = video_play.VideoReader()

if args.crop:
    encoder.set_cropstr(args.crop)

viewer_in = mp_image.MPImage(title='Original', width=200, height=200, auto_size=True)
viewer_out = mp_image.MPImage(title='Decoded', width=200, height=200, auto_size=True)

idx = 0
timestamp_ms = 0

enc_saved = open("enc.dat", "wb")

total_bytes = 0
total_frames = 0

while True:
    time.sleep(args.delay)
    img = cv2.imread(args.imgs[idx])
    cropped = encoder.crop_image(img)

    viewer_in.set_image(cropped)
    enc = encoder.add_image(img, timestamp_ms)

    timestamp_ms += 1000
    idx = (idx + 1) % len(args.imgs)

    if enc is None:
        continue
    enc_saved.write(enc)
    enc_saved.flush()

    encoder.report()

    total_bytes += len(enc)

    img = decoder.get_image(bytearray(enc))
    if img is None:
        continue
    while img is not None:
        total_frames += 1
        viewer_out.set_image(img)
        time.sleep(args.delay)
        img = decoder.get_image(bytearray(''))
    print("%u bytes/frame" % (total_bytes//total_frames))
