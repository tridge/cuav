#!/usr/bin/python

import numpy, os, time, threading, Queue, cv, sys

from cuav.image import scanner
from cuav.lib import cuav_util
from cuav.camera import chameleon

from optparse import OptionParser
parser = OptionParser("py_capture.py [options]")
parser.add_option("--depth", type='int', default=8, help="image depth")
parser.add_option("--mono", action='store_true', default=False, help="use mono camera")
parser.add_option("--save", action='store_true', default=False, help="save images in tmp/")
parser.add_option("--compress", action='store_true', default=False, help="compress images for saving")
parser.add_option("--scan", action='store_true', default=False, help="run the Joe scanner")
parser.add_option("--num-frames", "-n", type='int', default=0, help="number of images to capture")
parser.add_option("--scan-skip", type='int', default=0, help="number of scans to skip per image")
parser.add_option("--quality", type='int', default=95, help="compression quality")
parser.add_option("--brightness", type='int', default=100, help="auto-exposure brightness")
parser.add_option("--trigger", action='store_true', default=False, help="use triggering")
parser.add_option("--framerate", type='int', default=0, help="capture framerate Hz")
(opts, args) = parser.parse_args()

class capture_state():
  def __init__(self):
    self.save_thread = None
    self.scan_thread = None
    self.compress_thread = None
    self.bayer_thread = None
    self.bayer_queue = Queue.Queue()
    self.compress_queue = Queue.Queue()
    self.save_queue = Queue.Queue()
    self.scan_queue = Queue.Queue()

def start_thread(fn):
    '''start a thread running'''
    t = threading.Thread(target=fn)
    t.daemon = True
    t.start()
    return t

def get_base_time():
  '''we need to get a baseline time from the camera. To do that we trigger
  in single shot mode until we get a good image, and use the time we 
  triggered as the base time'''
  frame_time = None
  error_count = 0

  print('Opening camera')
  h = chameleon.open(not opts.mono, opts.depth, opts.brightness)
  print('camera is open')

  if opts.framerate != 0:
    chameleon.set_framerate(h, opts.framerate)

  while frame_time is None:
    try:
      base_time = time.time()
      im = numpy.zeros((960,1280),dtype='uint8' if opts.depth==8 else 'uint16')
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
        h = chameleon.open(not opts.mono, opts.depth, opts.brightness)
  return h, base_time, frame_time


def save_thread():
  '''thread for saving files'''
  try:
    os.mkdir('tmp')
  except Exception:
    pass
  while True:
    frame_time, im, is_jpeg = state.save_queue.get()
    if is_jpeg:
      filename = 'tmp/i%s.jpg' % cuav_util.frame_time(frame_time)
      chameleon.save_file(filename, im)
    else:
      filename = 'tmp/i%s.pgm' % cuav_util.frame_time(frame_time)
      chameleon.save_pgm(filename, im)

def bayer_thread():
  '''thread for debayering images'''
  while True:
    frame_time, im = state.bayer_queue.get()
    im_colour = numpy.zeros((960,1280,3),dtype='uint8')
    scanner.debayer(im, im_colour)
    if opts.compress:
      state.compress_queue.put((frame_time, im_colour))
    if opts.scan:
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.downsample(im_colour, im_640)
      state.scan_queue.put((frame_time, im_640))


def compress_thread():
  '''thread for compressing images'''
  while True:
    frame_time, im = state.compress_queue.get()
    jpeg = scanner.jpeg_compress(im, int(opts.quality))
    if opts.save:
      state.save_queue.put((frame_time, jpeg, True))

def scan_thread():
  '''thread for scanning for Joe'''
  total_time = 0
  count = 0
  while True:
    frame_time, im = state.scan_queue.get()
    t0=time.time()
    regions = scanner.scan(im)
    t1=time.time()
    total_time += (t1-t0)
    count += 1
    for i in range(opts.scan_skip):
      frame_time, im = state.scan_queue.get()
      

def run_capture():
  '''the main capture loop'''

  print("Getting base frame time")
  h, base_time, last_frame_time = get_base_time()

  if not opts.trigger:
    print('Starting continuous trigger')
    chameleon.trigger(h, True)
  
  frame_loss = 0
  num_captured = 0
  last_frame_counter = 0

  print('Starting main capture loop')

  while True:
    im = numpy.zeros((960,1280),dtype='uint8' if opts.depth==8 else 'uint16')
    try:
      if opts.trigger:
        chameleon.trigger(h, False)
      frame_time, frame_counter, shutter = chameleon.capture(h, 1000, im)
    except chameleon.error:
      print('failed to capture')
      continue
    if frame_time < last_frame_time:
      base_time += 128
    if last_frame_counter != 0:
      frame_loss += frame_counter - (last_frame_counter+1)

    if opts.compress or opts.scan:
      state.bayer_queue.put((base_time+frame_time, im))
    if opts.save and not opts.compress:
      state.save_queue.put((base_time+frame_time, im, False))

    print("Captured %s shutter=%f tdelta=%f(%.2f) ft=%f loss=%u qsave=%u qbayer=%u qcompress=%u scan=%u" % (
        cuav_util.frame_time(base_time+frame_time),
        shutter, 
        frame_time - last_frame_time,
        1.0/(frame_time - last_frame_time),
        frame_time,
        frame_loss,
        state.save_queue.qsize(),
        state.bayer_queue.qsize(),
        state.compress_queue.qsize(),
        state.scan_queue.qsize()))

    last_frame_time = frame_time
    last_frame_counter = frame_counter
    num_captured += 1
    if num_captured == opts.num_frames:
      break

  print('Closing camera')
  time.sleep(2)
  chameleon.close(h)

# main program
state = capture_state()

if opts.save:
  state.save_thread = start_thread(save_thread)

if opts.scan:
  state.scan_thread = start_thread(scan_thread)

if opts.scan or opts.compress:
  state.bayer_thread = start_thread(bayer_thread)

if opts.compress:
  state.compress_thread = start_thread(compress_thread)

run_capture()
