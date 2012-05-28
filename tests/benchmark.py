#!/usr/bin/python
'''
benchmark the base operations
'''

import numpy, os, time, cv, sys, math, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
import scanner, cuav_util, cuav_mosaic, mav_position, chameleon

from optparse import OptionParser
parser = OptionParser("benchmark.py [options] <filename>")
parser.add_option("--repeat", type='int', default=100, help="repeat count")
(opts, args) = parser.parse_args()

def process(filename):
  '''process one file'''
  pgm = cuav_util.PGM(filename)
  img_full_grey = pgm.array

  im_full = numpy.zeros((960,1280,3),dtype='uint8')
  im_640 = numpy.zeros((480,640,3),dtype='uint8')

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.debayer(img_full_grey, im_640)
  t1 = time.time()
  print('debayer: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.debayer_full(img_full_grey, im_full)
  t1 = time.time()
  print('debayer_full: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.downsample(im_full, im_640)
  t1 = time.time()
  print('downsample: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.scan(im_640)
  t1 = time.time()
  print('scan: %.1f fps' % (opts.repeat/(t1-t0)))

  for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    t0 = time.time()
    for i in range(opts.repeat):
      jpeg = scanner.jpeg_compress(im_full, quality)
    t1 = time.time()
    print('jpeg full quality %u: %.1f fps' % (quality, opts.repeat/(t1-t0)))

  for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    t0 = time.time()
    for i in range(opts.repeat):
      jpeg = scanner.jpeg_compress(im_640, quality)
    t1 = time.time()
    print('jpeg 640 quality %u: %.1f fps' % (quality, opts.repeat/(t1-t0)))

for f in args:
  process(f)
