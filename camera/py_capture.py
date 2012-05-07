#!/usr/bin/python

import chameleon, numpy, os, time

from optparse import OptionParser
parser = OptionParser("py_capture.py [options] <filename>")
parser.add_option("--depth", type='int', default=16, help="image depth")
parser.add_option("--mono", action='store_true', default=False, help="use mono camera")
parser.add_option("--save", action='store_true', default=False, help="save images in tmp/")
parser.add_option("--compress", action='store_true', default=False, help="compress images for saving")
parser.add_option("--scan", action='store_true', default=False, help="run the Joe scanner")
parser.add_option("--num-frames", "-n", type='int', default=0, help="number of images to capture")
(opts, args) = parser.parse_args()

class capture_state():
  def __init__(self):
    self.save_thread = None
    self.scan_thread = None
    self.compress_thread = None

def timestamp(frame_time):
    '''return a localtime timestamp with 0.01 second resolution'''
    hundredths = int(frame_time * 100.0) % 100
    return "%s%02u" % (time.strftime("%Y%m%d%H%M%S", time.localtime(frame_time)), hundredths)

def get_base_time(h):
  '''we need to get a baseline time from the camera. To do that we trigger
  in single shot mode until we get a good image, and use the time we 
  triggered as the base time'''
  frame_time = None
  while frame_time is None:
    try:
      base_time = time.time()
      im = numpy.zeros((960,1280),dtype='uint8' if opts.depth==8 else 'uint16')
      print('trigger')
      chameleon.trigger(h, False)
      print('capture')
      frame_time, frame_counter, shutter = chameleon.capture(h, im)
      base_time -= frame_time
    except chameleon.error:
      print('failed to capture')
  return base_time, frame_time

def run_capture():
  '''the main capture loop'''

  print('Opening camera')
  h = chameleon.open(not opts.mono, opts.depth)

  print("Getting base frame time")
  base_time, last_frame_time = get_base_time(h)

  print('Starting continuous trigger mode')
  chameleon.trigger(h, True)
  
  frame_loss = 0
  num_captured = 0
  last_frame_counter = 0

  while True:
    im = numpy.zeros((960,1280),dtype='uint8' if opts.depth==8 else 'uint16')
    try:
      frame_time, frame_counter, shutter = chameleon.capture(h, im)
    except chameleon.error:
      print('failed to capture')
      continue
    if frame_time < last_frame_time:
      base_time += 128
    filename = 'tmp/i%s.pgm' % timestamp(base_time + frame_time)
    if last_frame_counter != 0:
      frame_loss += frame_counter - (last_frame_counter+1)
    chameleon.save_pgm(h, filename, im)

    print("Captured to %s shutter=%f tdelta=%f ft=%f loss=%u" % (
        filename, shutter, 
        frame_time - last_frame_time,
        frame_time,
        frame_loss))

    last_frame_time = frame_time
    last_frame_counter = frame_counter
    num_captured += 1
    if num_captured == opts.num_frames:
      break

  print('Closing camera')
  chameleon.close(h)

# main program
run_capture()
