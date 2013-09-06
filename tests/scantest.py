#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob

from cuav.image import scanner
from cuav.lib import cuav_util, cuav_mosaic, mav_position, cuav_joe, cuav_region
from cuav.camera import chameleon, cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.mavproxy_map import mp_image

from optparse import OptionParser
parser = OptionParser("scantest.py [options] <directory>")
parser.add_option("--repeat", type='int', default=1, help="scan repeat count")
parser.add_option("--view", action='store_true', default=False, help="show images")
parser.add_option("--fullres", action='store_true', default=False, help="show full resolution")
parser.add_option("--gamma", type='int', default=0, help="gamma for 16 -> 8 conversion")
parser.add_option("--compress", action='store_true', default=False, help="compress images to *.jpg")
parser.add_option("--quality", type='int', default=95, help="jpeg compression quality")
parser.add_option("--mosaic", action='store_true', default=False, help="build a mosaic of regions")
parser.add_option("--mavlog", default=None, help="flight log for geo-referencing")
parser.add_option("--boundary", default=None, help="search boundary file")
parser.add_option("--max-deltat", default=0.0, type='float', help="max deltat for interpolation")
parser.add_option("--max-attitude", default=45, type='float', help="max attitude geo-referencing")
parser.add_option("--joe", default=None, help="file containing list of joe positions")
parser.add_option("--linkjoe", default=None, help="link joe images to this directory")
parser.add_option("--lens", default=4.0, type='float', help="lens focal length")
parser.add_option("--roll-stabilised", default=False, action='store_true', help="roll is stabilised")
parser.add_option("--gps-lag", default=0.0, type='float', help="GPS lag in seconds")
parser.add_option("--filter", default=False, action='store_true', help="filter using HSV")
parser.add_option("--minscore", default=3, type='int', help="minimum score")
parser.add_option("--altitude", type='int', default=None, help="camera assumed altitude")
parser.add_option("--filter-type", type='choice', default='simple', choices=['simple', 'compactness'], help="object filter type")
(opts, args) = parser.parse_args()

slipmap = None
mosaic = None

def process(args):
  '''process a set of files'''

  global slipmap, mosaic
  scan_count = 0
  files = []
  for a in args:
    if os.path.isdir(a):
      files.extend(glob.glob(os.path.join(a, '*.pgm')))
    else:
      files.append(a)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)
  region_count = 0
  joes = []

  if opts.mavlog:
    mpos = mav_position.MavInterpolator(gps_lag=opts.gps_lag)
    mpos.set_logfile(opts.mavlog)
  else:
    mpos = None

  if opts.boundary:
    boundary = cuav_util.polygon_load(opts.boundary)
  else:
    boundary = None

  if opts.mosaic:
    slipmap = mp_slipmap.MPSlipMap(service='GoogleSat', elevation=True, title='Map')
    icon = slipmap.icon('planetracker.png')
    slipmap.add_object(mp_slipmap.SlipIcon('plane', (0,0), icon, layer=3, rotation=0,
                                           follow=True,
                                           trail=mp_slipmap.SlipTrail()))
    C_params = cam_params.CameraParams(lens=opts.lens)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..',
                        'cuav', 'data', 'chameleon1_arecont0.json')
    C_params.load(path)
    mosaic = cuav_mosaic.Mosaic(slipmap, C=C_params)
    if boundary is not None:
      mosaic.set_boundary(boundary)

  if opts.joe:
    joes = cuav_util.polygon_load(opts.joe)
    if boundary:
      for i in range(len(joes)):
        joe = joes[i]
        if cuav_util.polygon_outside(joe, boundary):
          print("Error: joe outside boundary", joe)
          return
        icon = slipmap.icon('flag.png')
        slipmap.add_object(mp_slipmap.SlipIcon('joe%u' % i, (joe[0],joe[1]), icon, layer=4))

  joelog = cuav_joe.JoeLog('joe.log')      

  if opts.view:
    viewer = mp_image.MPImage(title='Image')

  frame_time = 0

  for f in files:
    if mpos:
      frame_time = cuav_util.parse_frame_time(f)
      try:
        if opts.roll_stabilised:
          roll = 0
        else:
          roll = None
        pos = mpos.position(frame_time, opts.max_deltat,roll=roll)
        slipmap.set_position('plane', (pos.lat, pos.lon), rotation=pos.yaw)
      except mav_position.MavInterpolatorException as e:
        print e
        pos = None
    else:
      pos = None

    # check for any events from the map
    if opts.mosaic:
      slipmap.check_events()
      mosaic.check_events()

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
      im_orig = cv.LoadImage(f)
      (w,h) = cuav_util.image_shape(im_orig)
      if (w,h) != (1280,960):
        im_full = cv.CreateImage((1280, 960), 8, 3)
        cv.Resize(im_orig, im_full)
        cv.ConvertScale(im_full, im_full, scale=0.3)
      else:
        im_full = im_orig
      im_640 = cv.CreateImage((640, 480), 8, 3)
      cv.Resize(im_full, im_640, cv.CV_INTER_NN)
      im_640 = numpy.ascontiguousarray(cv.GetMat(im_640))
      im_full = numpy.ascontiguousarray(cv.GetMat(im_full))

    count = 0
    total_time = 0
    img_scan = im_640

    t0=time.time()
    for i in range(opts.repeat):
      if opts.fullres:
        regions = scanner.scan_full(im_full)
        regions = cuav_region.RegionsConvert(regions, 1280, 960)
      else:
        regions = scanner.scan(img_scan)
        regions = cuav_region.RegionsConvert(regions)
      count += 1
    t1=time.time()

    if opts.filter:
      regions = cuav_region.filter_regions(im_full, regions, frame_time=frame_time, min_score=opts.minscore,
                                           filter_type=opts.filter_type)

    scan_count += 1

    # optionally link all the images with joe into a separate directory
    # for faster re-running of the test with just joe images
    if pos and opts.linkjoe and len(regions) > 0:
      cuav_util.mkdir_p(opts.linkjoe)
      if not cuav_util.polygon_outside((pos.lat, pos.lon), boundary):
        joepath = os.path.join(opts.linkjoe, os.path.basename(f))
        if os.path.exists(joepath):
          os.unlink(joepath)
        os.symlink(f, joepath)

    if pos and len(regions) > 0:
      joelog.add_regions(frame_time, regions, pos, f, width=1280, height=960, altitude=opts.altitude)

      if boundary:
        regions = cuav_region.filter_boundary(regions, boundary, pos)

    region_count += len(regions)

    if opts.mosaic and len(regions) > 0:
      composite = cuav_mosaic.CompositeThumbnail(cv.GetImage(cv.fromarray(im_full)), regions)
      thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
      mosaic.add_regions(regions, thumbs, f, pos)

    if opts.compress:
      jpeg = scanner.jpeg_compress(im_full, opts.quality)
      jpeg_filename = f[:-4] + '.jpg'
      if os.path.exists(jpeg_filename):
        print('jpeg %s already exists' % jpeg_filename)
        continue
      chameleon.save_file(jpeg_filename, jpeg)

    if opts.view:
      if opts.fullres:
        img_view = im_full
      else:
        img_view = img_scan
      mat = cv.fromarray(img_view)
      for r in regions:
        (x1,y1,x2,y2) = r.tuple()
        (w,h) = cuav_util.image_shape(img_view)
        x1 = x1*w//1280
        x2 = x2*w//1280
        y1 = y1*h//960
        y2 = y2*h//960
        cv.Rectangle(mat, (max(x1-2,0),max(y1-2,0)), (x2+2,y2+2), (255,0,0), 2)
      cv.CvtColor(mat, mat, cv.CV_BGR2RGB)
      viewer.set_image(mat)

    total_time += (t1-t0)
    if t1 != t0:
      print('%s scan %.1f fps  %u regions [%u/%u]' % (
        f, count/total_time, region_count, scan_count, num_files))


# main program

process(args)
while True:
  if opts.mosaic:
    slipmap.check_events()
    mosaic.check_events()
  time.sleep(0.01)
  
