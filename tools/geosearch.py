#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob

from cuav.lib import cuav_util
from cuav.image import scanner
from cuav.lib import cuav_mosaic, mav_position, cuav_joe, cuav_region
from cuav.camera import cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.mavproxy_map import mp_image

from optparse import OptionParser
parser = OptionParser("geosearch.py [options] <directory>")
parser.add_option("--view", action='store_true', default=False, help="show images")
parser.add_option("--filter", default=True, action='store_true', help="filter using HSV")
parser.add_option("--minscore", default=500, type='int', help="minimum score")
parser.add_option("--split", default=1, type='int', help="split in X/Y direction by this factor")
parser.add_option("--grid", action='store_true', default=False, help="add a UTM grid")
parser.add_option("--mission", default=[], action='append', help="show mission")
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
      files.extend(glob.glob(os.path.join(a, '*.jpg')))
    else:
      if a.find('*') != -1:
        files.extend(glob.glob(a))
      else:
        files.append(a)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)
  region_count = 0

  slipmap = mp_slipmap.MPSlipMap(service='GoogleSat', elevation=True, title='Map')
  icon = slipmap.icon('planetracker.png')
  slipmap.add_object(mp_slipmap.SlipIcon('plane', (0,0), icon, layer=3, rotation=0,
                                         follow=True,
                                         trail=mp_slipmap.SlipTrail()))

  if opts.grid:
    slipmap.add_object(mp_slipmap.SlipGrid('grid', layer=1, linewidth=1, colour=(255,255,0)))

  if opts.mission:
    from pymavlink import mavwp
    for file in opts.mission:
      wp = mavwp.MAVWPLoader()
      wp.load(file)
      boundary = wp.polygon()
      slipmap.add_object(mp_slipmap.SlipPolygon('mission-%s' % file, boundary, layer=1, linewidth=1, colour=(255,255,255)))


  mosaic = cuav_mosaic.Mosaic(slipmap)

  joelog = cuav_joe.JoeLog('joe.log')      

  if opts.view:
    viewer = mp_image.MPImage(title='Image')

  for f in files:
      pos = mav_position.exif_position(f)
      slipmap.set_position('plane', (pos.lat, pos.lon), rotation=pos.yaw)

      # check for any events from the map
      slipmap.check_events()
      mosaic.check_events()

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
      regions = scanner.scan(img_scan)
      count += 1
      regions = cuav_region.RegionsConvert(regions)
      t1=time.time()

      frame_time = pos.time

      if opts.filter:
          regions = cuav_region.filter_regions(im_full, regions, frame_time=frame_time, min_score=opts.minscore)

      scan_count += 1

      mosaic.add_image(pos.time, f, pos)

      if pos and len(regions) > 0:
          joelog.add_regions(frame_time, regions, pos, f, width=1280, height=960, altitude=0)

      region_count += len(regions)

      if len(regions) > 0:
          composite = cuav_mosaic.CompositeThumbnail(cv.GetImage(cv.fromarray(im_full)), regions,
                                                     quality=80)
          j = open('composite.jpg', mode='wb')
          j.write(composite)
          j.close()
          thumbs = cuav_mosaic.ExtractThumbs(cv.LoadImage('composite.jpg'), len(regions))
          mosaic.add_regions(regions, thumbs, f, pos)

      if opts.view:
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
              os.path.basename(f), count/total_time, region_count, scan_count, num_files))


if __name__ == '__main__':
  # main program
  process(args)
  while True:
    slipmap.check_events()
    mosaic.check_events()
    time.sleep(0.002)
