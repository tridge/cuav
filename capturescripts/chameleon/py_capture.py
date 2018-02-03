#!/usr/bin/python
'''
Capture images from a Ptgrey Chameleon camera, with precise timestamps
'''

import future
import numpy, os, time, threading, cv2, sys, argparse
from queue import Queue
import chameleon

parser = argparse.ArgumentParser("py_capture.py [options]")
parser.add_argument("--depth", type=int, default=8, help="image depth")
parser.add_argument("--mono", action='store_true', default=False, help="use mono camera")
parser.add_argument("--save", action='store', default=None, help="save images in the specified folder")
parser.add_argument("--format", default='jpg', choices=['png', 'jpg', 'pgm'], help="Output format")
parser.add_argument("--num-frames", "-n", type=int, default=0, help="number of images to capture")
parser.add_argument("--scan-skip", type=int, default=0, help="number of scans to skip per image")
parser.add_argument("--brightness", type=int, default=100, help="auto-exposure brightness")
parser.add_argument("--trigger", action='store_true', default=False, help="use triggering")
parser.add_argument("--framerate", type=int, default=0, help="capture framerate Hz")
parser.add_argument("--reduction", type=int, default=0, help="frame reduction factor")
parser.add_argument("--make-fake", action='store', default=None, help="path/file for symlinked current image")
opts = parser.parse_args()

class capture_state():
  def __init__(self):
    self.save_thread = None
    self.bayer_thread = None
    self.bayer_queue = Queue()
    self.save_queue = Queue()

def start_thread(fn):
    '''start a thread running'''
    t = threading.Thread(target=fn)
    t.daemon = True
    t.start()
    return t

def get_frame_time(t):
    '''return a time string for a filename with 0.01 sec resolution'''
    # round to the nearest 100th of a second
    t += 0.005
    hundredths = int(t * 100.0) % 100
    return "{0}{1}Z".format(time.strftime("%Y%m%d%H%M%S", time.gmtime(t)), hundredths)

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
        if opts.framerate != 0:
          chameleon.set_framerate(h, opts.framerate)
  return h, base_time, frame_time


def save_thread():
  '''thread for saving files'''
  try:
    os.mkdir(opts.save)
  except OSError:
    pass

  last_filename = None
  while True:
    frame_time, im = state.save_queue.get()
    filename =  os.path.join(opts.save, '{0}.{1}'.format(get_frame_time(frame_time), opts.format))
    if opts.format == 'jpg':
        cv2.imwrite(filename, im, [cv2.IMWRITE_JPEG_QUALITY, 99])
    elif opts.format == 'png':
        cv2.imwrite(filename, im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    else:
        cv2.imwrite(filename, im)
    if opts.make_fake is not None:
      try:
        os.unlink(opts.make_fake)
      except OSError:
        pass
      os.symlink(filename, opts.make_fake)
      if last_filename is not None:
        try:
          os.unlink(last_filename)
        except OSError:
          pass
      last_filename = filename

def bayer_thread():
  '''thread for debayering images'''
  while True:
    frame_time, im = state.bayer_queue.get()
    im_colour = cv2.cvtColor(im, cv2.COLOR_BAYER_GR2BGR)
    state.save_queue.put((frame_time, im_colour))

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
  error_count = 0
  
  print('Starting main capture loop')

  while True:
    im = numpy.zeros((960,1280),dtype='uint8' if opts.depth==8 else 'uint16')
    try:
      if opts.trigger:
        chameleon.trigger(h, False)
      frame_time, frame_counter, shutter = chameleon.capture(h, 1000, im)
    except chameleon.error:
      print('failed to capture')
      error_count += 1
      if error_count > 3:
        error_count = 0
        print('re-opening camera')
        chameleon.close(h)
        h = chameleon.open(not opts.mono, opts.depth, opts.brightness)
        if opts.framerate != 0:
          chameleon.set_framerate(h, opts.framerate)
        if not opts.trigger:
          print('Starting continuous trigger')
          chameleon.trigger(h, True)
      continue
    if frame_time < last_frame_time:
      base_time += 128
    if last_frame_counter != 0:
      frame_loss += frame_counter - (last_frame_counter+1)

    if opts.format != 'pgm':
      state.bayer_queue.put((base_time+frame_time, im))
    if opts.save and opts.format == 'pgm':
      if opts.reduction == 0 or num_captured % opts.reduction == 0:
        state.save_queue.put((base_time+frame_time, im))

    print("Captured %s shutter=%f tdelta=%f(%.2f) ft=%f loss=%u qsave=%u qbayer=%u" % (
        get_frame_time(base_time+frame_time),
        shutter, 
        frame_time - last_frame_time,
        1.0/(frame_time - last_frame_time),
        frame_time,
        frame_loss,
        state.save_queue.qsize(),
        state.bayer_queue.qsize()))

    error_count = 0
    last_frame_time = frame_time
    last_frame_counter = frame_counter
    num_captured += 1
    if num_captured == opts.num_frames:
      break

  print('Closing camera')
  time.sleep(2)
  chameleon.close(h)

if __name__ == '__main__':
    state = capture_state()

    if opts.save:
      state.save_thread = start_thread(save_thread)
      state.bayer_thread = start_thread(bayer_thread)

    run_capture()
