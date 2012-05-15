#!/usr/bin/python

import chameleon, numpy, os, time, cv, sys, util

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
import scanner

from optparse import OptionParser
parser = OptionParser("scantest.py [options] <filename..>")
parser.add_option("--repeat", type='int', default=1, help="scan repeat count")
parser.add_option("--view", action='store_true', default=False, help="show images")
parser.add_option("--fullres", action='store_true', default=False, help="debayer at full resolution")
(opts, args) = parser.parse_args()

class state():
  def __init__(self):
    pass

def process(files):
  '''process a set of files'''

  for f in files:
    stat = os.stat(f)
    pgm = util.PGM(f)
    im = pgm.array
    if opts.fullres:
      im_8bit = numpy.zeros((960,1280,1),dtype='uint8')
      scanner.reduce_depth(im, im_8bit)
      im_colour = numpy.zeros((960,1280,3),dtype='uint8')
      scanner.debayer_16_full(im, im_colour)
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.downsample(im_colour, im_640)
    else:
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.debayer(im, im_640)

    count = 0
    region_count = 0
    total_time = 0

    t0=time.time()
    for i in range(opts.repeat):
      regions = scanner.scan(im_640)
      count += 1
    t1=time.time()
    region_count += len(regions)

    if opts.view:
      mat = cv.fromarray(im_640)
      for (x1,y1,x2,y2) in regions:
        cv.Rectangle(mat, (x1,y1), (x2,y2), (255,0,0), 1)
      cv.ShowImage('Viewer', mat)
      cv.WaitKey(1)
      cv.WaitKey(1)

    total_time += (t1-t0)
    print('%s scan %f fps  %u regions' % (f, count/total_time, region_count))
    

# main program
state = state()

process(args)
cv.WaitKey(2000)
cv.DestroyWindow('Viewer')
