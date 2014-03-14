#!/usr/bin/python

import cv, time, sys, os, numpy, threading, Queue
from cuav.image import scanner
from cuav.lib import cuav_util
from cuav.camera import chameleon

from optparse import OptionParser
parser = OptionParser("py_viewer.py [options]")
parser.add_option("--depth", type='int', default=8, help="capture depth")
parser.add_option("--output", default='tmp', help="output directory")
parser.add_option("--half", action='store_true', default=False, help="show half res")
parser.add_option("--brightness", type='float', default=1.0, help="display brightness")
parser.add_option("--capbrightness", type='int', default=150, help="capture brightness in camera")
parser.add_option("--gamma", type='int', default=1024, help="set gamma (if 16 bit images)")
(opts, args) = parser.parse_args()

def start_thread(fn):
    '''start a thread running'''
    t = threading.Thread(target=fn)
    t.daemon = True
    t.start()
    return t


def get_base_time(depth=8, colour=1, capture_brightness=150):
  '''we need to get a baseline time from the camera. To do that we trigger
  in single shot mode until we get a good image, and use the time we 
  triggered as the base time'''
  frame_time = None
  error_count = 0

  print('Opening camera')
  h = chameleon.open(colour, depth, capture_brightness)

  print('Getting camare base_time')
  while frame_time is None:
    try:
      im = numpy.zeros((960,1280),dtype='uint8' if depth==8 else 'uint16')
      base_time = time.time()
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
        h = chameleon.open(colour, depth, capture_brightness)
  print('base_time=%f' % base_time)
  return h, base_time, frame_time

save_queue = Queue.Queue()

def save_thread():
  global save_queue
  while True:
    (frame_time, im) = save_queue.get()
    filename = '%s/%s.pgm' % (opts.output, cuav_util.frame_time(frame_time))
    chameleon.save_pgm(filename, im)

colour = 1
h, base_time, last_frame_time = get_base_time(opts.depth, colour=colour, capture_brightness=opts.capbrightness)
print("Found camera: colour=%u GUID=%x" % (colour, chameleon.guid(h)))
if opts.depth == 8:
  dtype = 'uint8'
else:
  dtype = 'uint16'
im = numpy.zeros((960,1280),dtype=dtype)

cv.NamedWindow('Viewer')

tstart = time.time()

chameleon.trigger(h, True)
chameleon.set_gamma(h, opts.gamma)

cuav_util.mkdir_p(opts.output)

start_thread(save_thread)

i=0
lost = 0
while True:
  try:
    frame_time, frame_counter, shutter = chameleon.capture(h, 300, im)
  except chameleon.error, msg:
    lost += 1
    continue
  if frame_time < last_frame_time:
    base_time += 128
  save_queue.put((frame_time+base_time, im))
  last_frame_time = frame_time
  img_colour = numpy.zeros((960,1280,3),dtype='uint8')
  scanner.debayer(im, img_colour)
  img_colour = cv.GetImage(cv.fromarray(img_colour))
  if opts.half:
    img_640 = cv.CreateImage((640,480), 8, 3)
    cv.Resize(img_colour, img_640)
    img_colour = img_640

  cv.ConvertScale(img_colour, img_colour, scale=opts.brightness)
  cv.ShowImage('Viewer', img_colour)
  key = cv.WaitKey(1)
  i += 1

  if i % 10 == 0:
    tdiff = time.time() - tstart
    print("captured %u  lost %u  %.1f fps shutter=%f" % (
      i, lost,
      10/tdiff, shutter));
    tstart = time.time()

  key = cv.WaitKey(1)
  if key == -1:
    continue
  if key == ord('q'):
    break

chameleon.close(h)
cv.DestroyWindow('Viewer')

