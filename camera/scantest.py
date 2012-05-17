#!/usr/bin/python

import chameleon, numpy, os, time, cv, sys, util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

from optparse import OptionParser
parser = OptionParser("scantest.py [options] <filename..>")
parser.add_option("--repeat", type='int', default=1, help="scan repeat count")
parser.add_option("--view", action='store_true', default=False, help="show images")
parser.add_option("--fullres", action='store_true', default=False, help="debayer at full resolution")
parser.add_option("--gamma", type='int', default=0, help="gamma for 16 -> 8 conversion")
parser.add_option("--yuv", action='store_true', default=False, help="use YUV conversion")
parser.add_option("--mosaic", action='store_true', default=False, help="build a mosaic of regions")
(opts, args) = parser.parse_args()

class state():
  def __init__(self):
    pass

class Mosaic():
  '''keep a mosaic of found regions'''
  def __init__(self, width=512, height=512):
    self.width = width
    self.height = height
    self.mosaic = numpy.zeros((width,height,3),dtype='uint8')
    self.region_index = 0

  def add_regions(self, regions, img):
    '''add some regions'''
    for (x1,y1,x2,y2) in regions:
      dest_x = (self.region_index * 32) % self.height
      dest_y = ((self.region_index * 32) / self.width) * 32
      midx = (x1+x2)/2
      midy = (y1+y2)/2
      for x in range(-16, 16):
        for y in range(-16, 16):
          if (y+midy < 0 or x+midx < 0 or
              y+midy >= img.shape[0] or x+midx >= img.shape[1]):
            continue
          px = img[y+midy, x+midx]
          self.mosaic[dest_y+y+16, dest_x+x+16] = px
      self.region_index += 1
      if self.region_index >= (self.width/32)*(self.height/32):
        self.region_index = 0
    

def update_mosaic(mosaic, regions):
  '''add to the mosaic'''
  pass

def process(files):
  '''process a set of files'''

  scan_count = 0
  num_files = len(files)

  if opts.mosaic:
    mosaic = Mosaic()

  for f in files:
    stat = os.stat(f)
    pgm = util.PGM(f)
    im = pgm.array
    if opts.fullres:
      if pgm.eightbit:
        im_8bit = im
      else:
        im_8bit = numpy.zeros((960,1280,1),dtype='uint8')
        if opts.gamma != 0:
          scanner.gamma_correct(im, im_8bit, opts.gamma)
        else:
          scanner.reduce_depth(im, im_8bit)
      im_colour = numpy.zeros((960,1280,3),dtype='uint8')
      scanner.debayer_full(im_8bit, im_colour)
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.downsample(im_colour, im_640)
    else:
      if opts.gamma != 0:
        im_8bit = numpy.zeros((960,1280,1),dtype='uint8')
        scanner.gamma_correct(im, im_8bit, opts.gamma)
        im = im_8bit
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.debayer(im, im_640)

    count = 0
    region_count = 0
    total_time = 0

    if opts.yuv:
      img_scan = numpy.zeros((480,640,3),dtype='uint8')
      scanner.rgb_to_yuv(im_640, img_scan)
    else:
      img_scan = im_640

    t0=time.time()
    for i in range(opts.repeat):
      regions = scanner.scan(img_scan)
      count += 1
    t1=time.time()
    region_count += len(regions)
    scan_count += 1

    if opts.mosaic:
      mosaic.add_regions(regions, img_scan)
      mat = cv.fromarray(mosaic.mosaic)
      cv.ShowImage('Mosaic', mat)
    
    if opts.view:
      mat = cv.fromarray(img_scan)
      for (x1,y1,x2,y2) in regions:
        cv.Rectangle(mat, (x1,y1), (x2,y2), (255,0,0), 1)
      cv.ShowImage('Viewer', mat)
      cv.WaitKey(1)
      cv.WaitKey(1)

    total_time += (t1-t0)
    print('%s scan %f fps  %u regions [%u/%u]' % (
      f, count/total_time, region_count, scan_count, num_files))
    

# main program
state = state()

process(args)
cv.WaitKey()
cv.DestroyWindow('Viewer')
