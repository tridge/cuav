#!/usr/bin/python
'''
benchmark the base operations
'''

import numpy, os, time, cv, sys, math, sys, cPickle, pickle

from cuav.image import scanner
from cuav.lib import cuav_util, mav_position

from optparse import OptionParser
parser = OptionParser("benchmark.py [options] <filename>")
parser.add_option("--repeat", type='int', default=100, help="repeat count")
(opts, args) = parser.parse_args()

class ImagePacket:
    '''a jpeg image sent to the ground station'''
    def __init__(self, frame_time, jpeg):
        self.frame_time = frame_time
        self.jpeg = jpeg

def process(filename):
  '''process one file'''
  pgm = cuav_util.PGM(filename)
  img_full_grey = pgm.array

  im_full = numpy.zeros((960,1280,3),dtype='uint8')
  im_640 = numpy.zeros((480,640,3),dtype='uint8')

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.debayer_half(img_full_grey, im_640)
  t1 = time.time()
  print('debayer: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.debayer(img_full_grey, im_full)
  t1 = time.time()
  print('debayer_full: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  im_full2 = cv.CreateImage((1280,960),8,3)
  img_full_grey2 = cv.GetImage(cv.fromarray(img_full_grey)) 
  for i in range(opts.repeat):
      cv.CvtColor(img_full_grey2, im_full2, cv.CV_BayerBG2BGR)
  t1 = time.time()
  print('debayer_cv_full: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
      img = cv.GetImage(cv.fromarray(im_full))
      cv.CvtColor(img, img, cv.CV_RGB2HSV)
  t1 = time.time()
  print('RGB2HSV_full: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
      img = cv.GetImage(cv.fromarray(im_640))
      cv.CvtColor(img, img, cv.CV_RGB2HSV)
  t1 = time.time()
  print('RGB2HSV_640: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
      thumb = numpy.empty((100,100,3),dtype='uint8')
      scanner.rect_extract(im_full, thumb, 120, 125)
  t1 = time.time()
  print('rect_extract: %.1f fps' % (opts.repeat/(t1-t0)))

  t0 = time.time()
  for i in range(opts.repeat):
      thumb = cuav_util.SubImage(cv.GetImage(cv.fromarray(im_full)), (120,125,100,100))
  t1 = time.time()
  print('SubImage: %.1f fps' % (opts.repeat/(t1-t0)))


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

  t0 = time.time()
  for i in range(opts.repeat):
    scanner.scan(im_full)
  t1 = time.time()
  print('scan_full: %.1f fps' % (opts.repeat/(t1-t0)))

  if not hasattr(scanner, 'jpeg_compress'):
      return
  
  for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    t0 = time.time()
    for i in range(opts.repeat):
      jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(im_full, quality)),
                           protocol=cPickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print('jpeg full quality %u: %.1f fps  %u bytes' % (quality, opts.repeat/(t1-t0), len(bytes(jpeg))))

  for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    t0 = time.time()
    for i in range(opts.repeat):
      img2 = cv.fromarray(im_full)
      jpeg = cPickle.dumps(ImagePacket(time.time(), 
                                       cv.EncodeImage('.jpeg', img2, [cv.CV_IMWRITE_JPEG_QUALITY,quality]).tostring()),
                           protocol=cPickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print('EncodeImage full quality %u: %.1f fps  %u bytes' % (quality, opts.repeat/(t1-t0), len(bytes(jpeg))))

  for quality in [30, 40, 50, 60, 70, 80, 90, 95]:
    t0 = time.time()
    for i in range(opts.repeat):
      jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(im_640, quality)),
                           protocol=cPickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print('jpeg 640 quality %u: %.1f fps  %u bytes' % (quality, opts.repeat/(t1-t0), len(bytes(jpeg))))

  for thumb_size in [10, 20, 40, 60, 80, 100]:
    thumb = numpy.zeros((thumb_size,thumb_size,3),dtype='uint8')
    t0 = time.time()
    for i in range(opts.repeat):
      scanner.rect_extract(im_full, thumb, 0, 0)
      jpeg = cPickle.dumps(ImagePacket(time.time(), scanner.jpeg_compress(thumb, 85)),
                           protocol=cPickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print('thumb %u quality 85: %.1f fps  %u bytes' % (thumb_size, opts.repeat/(t1-t0), len(bytes(jpeg))))
    

for f in args:
  process(f)
