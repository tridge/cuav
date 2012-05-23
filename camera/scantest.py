#!/usr/bin/python

import chameleon, numpy, os, time, cv, sys, math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'lib'))
import scanner, cuav_util, cuav_mosaic, mav_position, chameleon

from optparse import OptionParser
parser = OptionParser("scantest.py [options] <filename..>")
parser.add_option("--repeat", type='int', default=1, help="scan repeat count")
parser.add_option("--view", action='store_true', default=False, help="show images")
parser.add_option("--fullres", action='store_true', default=False, help="show full resolution")
parser.add_option("--gamma", type='int', default=0, help="gamma for 16 -> 8 conversion")
parser.add_option("--yuv", action='store_true', default=False, help="use YUV conversion")
parser.add_option("--compress", action='store_true', default=False, help="show jpeg compressed images")
parser.add_option("--quality", type='int', default=80, help="jpeg compression quality")
parser.add_option("--mosaic", action='store_true', default=False, help="build a mosaic of regions")
parser.add_option("--mavlog", default=None, help="flight log for geo-referencing")
parser.add_option("--boundary", default=None, help="search boundary file")
parser.add_option("--max-deltat", default=1.0, type='float', help="max deltat for interpolation")
parser.add_option("--max-attitude", default=45, type='float', help="max attitude geo-referencing")
parser.add_option("--fill-map", default=False, action='store_true', help="show all images on map")
(opts, args) = parser.parse_args()

class state():
  def __init__(self):
    pass

def process(files):
  '''process a set of files'''

  scan_count = 0
  num_files = len(files)
  region_count = 0

  if opts.mavlog:
    mpos = mav_position.MavInterpolator()
    mpos.set_logfile(opts.mavlog)
  else:
    mpos = None

  if opts.boundary:
    boundary = cuav_util.polygon_load(opts.boundary)
  else:
    boundary = None

  if opts.mosaic:
    mosaic = cuav_mosaic.Mosaic()
    if boundary is not None:
      mosaic.set_boundary(boundary)
    if opts.fill_map:
      mosaic.fill_map = True

  for f in files:
    frame_time = cuav_util.parse_frame_time(f)
    if mpos:
      try:
        pos = mpos.position(frame_time, opts.max_deltat)
      except mav_position.MavInterpolatorDeltaTException:
        pos = None
    else:
      pos = None
    if f.endswith('.pgm'):
      pgm = cuav_util.PGM(f)
      im = pgm.array
      if pgm.eightbit:
        im_8bit = im
      else:
        im_8bit = numpy.zeros((960,1280,1),dtype='uint8')
        if opts.gamma != 0:
          scanner.gamma_correct(im, im_8bit, opts.gamma)
        else:
          scanner.reduce_depth(im, im_8bit)
      im_full = numpy.zeros((960,1280,3),dtype='uint8')
      scanner.debayer_full(im_8bit, im_full)
      im_640 = numpy.zeros((480,640,3),dtype='uint8')
      scanner.downsample(im_full, im_640)
    else:
      im_full = cv.LoadImage(f)
      im_640 = cv.CreateImage((640, 480), 8, 3)
      cv.Resize(im_full, im_640)
      im_640 = numpy.ascontiguousarray(cv.GetMat(im_640))

    count = 0
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
      mosaic.add_regions(regions, img_scan, f, pos)
      if (opts.fill_map and pos and
          math.fabs(pos.roll) < opts.max_attitude and
          math.fabs(pos.pitch) < opts.max_attitude):
        mosaic.add_image(img_scan, pos)
    
    if opts.view:
      if opts.fullres:
        img_view = im_full
      else:
        img_view = img_scan
      if opts.compress:
        jpeg = scanner.jpeg_compress(img_view, opts.quality)
        chameleon.save_file('view.jpg', jpeg)
        mat = cv.LoadImage('view.jpg')
      else:
        mat = cv.fromarray(img_view)
      for (x1,y1,x2,y2) in regions:
        if opts.fullres:
          x1 *= 2
          y1 *= 2
          x2 *= 2
          y2 *= 2
        cv.Rectangle(mat, (x1,y1), (x2,y2), (255,0,0), 1)
      cv.ShowImage('Viewer', mat)
      cv.WaitKey(1)
      cv.WaitKey(1)

    total_time += (t1-t0)
    print('%s scan %f fps  %u regions [%u/%u]' % (
      f, count/total_time, region_count, scan_count, num_files))
    

if opts.view:
    cv.NamedWindow('Viewer')

# main program
state = state()

process(args)
cuav_util.cv_wait_quit()
cv.DestroyWindow('Viewer')
