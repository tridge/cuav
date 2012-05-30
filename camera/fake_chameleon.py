#!/usr/bin/env
'''
emulate a chameleon camera, getting images from a playback tool
'''

import chameleon, time, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
import cuav_util

error = chameleon.error
continuous_mode = False
fake = 'fake_chameleon.pgm'
frame_counter = 0
trigger_time = 0
frame_rate = 7.5
chameleon_gamma = 950
last_frame_time = 0

def open(colour, depth, brightness):
    return 0

def trigger(h, continuous):
    global continuous_mode, trigger_time
    continuous_mode = continuous
    trigger_time = time.time()

def capture(h, timeout, img):
    global continuous_mode, trigger_time, frame_rate, frame_counter, fake, last_frame_time
    tnow = time.time()
    due = trigger_time + (1.0/frame_rate)
    if tnow < due:
        time.sleep(due - tnow)
        timeout -= int(due*1000)
    # wait for a new image to appear
    frame_time = cuav_util.parse_frame_time(os.path.realpath(fake))
    while frame_time == last_frame_time and timeout > 0:
        timeout -= 10
        time.sleep(0.01)
        frame_time = cuav_util.parse_frame_time(os.path.realpath(fake))

    if last_frame_time == frame_time:
        raise chameleon.error("timeout waiting for fake image")
    last_frame_time = frame_time
    try:
        fake_img = cuav_util.PGM(fake)
    except Exception, msg:
        print msg
        raise chameleon.error('missing %s' % fake)
    frame_counter += 1
    img.data = fake_img.array.data
    if continuous_mode:
        trigger_time = time.time()
    return frame_time, frame_counter, 0

def close(h):
    return

def set_gamma(h, gamma):
    global chameleon_gamma
    chameleon_gamma = gamma

def save_pgm(filename, img):
    return chameleon.save_pgm(filename, img)

def save_file(filename, bytes):
    return chameleon.save_file(filename, bytes)
