#!/usr/bin/python

import numpy, os, time, cv, sys, math, sys, glob
import multiprocessing

from cuav.lib import cuav_util
from cuav.image import scanner
from cuav.lib import cuav_mosaic, mav_position, cuav_joe, cuav_region
from cuav.camera import cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.lib import mp_image
from MAVProxy.modules.lib.mp_settings import MPSettings, MPSetting

slipmap = None
mosaic = None

def file_list(directory, extensions):
  '''return file list for a directory'''
  flist = []
  for (root, dirs, files) in os.walk(directory):
    for f in files:
      extension = f.split('.')[-1]
      if extension.lower() in extensions:
        flist.append(os.path.join(root, f))
  return flist


def parse_gamma_log(gammalog):
  '''parse gamma.log to process a mapping between frame_time string and GMT time of capture'''
  f = open(gammalog)
  lines = f.readlines()
  f.close()
  ret = {}
  for line in lines:
    a = line.split()
    capture_time = float(a[2])
    tstring = a[3]
    ret[tstring] = capture_time
  return ret

def parse_gamma_time(fname, gamma):
  '''get GMT capture_time from filename and gamma hash'''
  (root, ext) = os.path.splitext(os.path.basename(fname))
  if root.lower().startswith("raw"):
    root = root[3:]
  if root in gamma:
    return gamma[root]
  return cuav_util.parse_frame_time(fname)

def process(args):
  '''process a set of files'''

  global slipmap, mosaic
  scan_count = 0
  files = []
  for a in args:
    if os.path.isdir(a):
      files.extend(file_list(a, ['jpg', 'pgm', 'png']))
    else:
      if a.find('*') != -1:
        files.extend(glob.glob(a))
      else:
        files.append(a)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)
  region_count = 0

  slipmap = mp_slipmap.MPSlipMap(service=opts.service, elevation=True, title='Map')
  icon = slipmap.icon('redplane.png')
  slipmap.add_object(mp_slipmap.SlipIcon('plane', (0,0), icon, layer=3, rotation=0,
                                         follow=True,
                                         trail=mp_slipmap.SlipTrail()))

  if opts.mission:
    from pymavlink import mavwp
    wp = mavwp.MAVWPLoader()
    wp.load(opts.mission)
    boundary = wp.polygon()
    slipmap.add_object(mp_slipmap.SlipPolygon('mission', boundary, layer=1,
                                              linewidth=1, colour=(255,255,255)))


  if opts.mavlog:
    mpos = mav_position.MavInterpolator()
    mpos.set_logfile(opts.mavlog)
  else:
    mpos = None

  if opts.gammalog is not None:
    gamma = parse_gamma_log(opts.gammalog)
  else:
    gamma = None

  if opts.kmzlog:
    kmzpos = mav_position.KmlPosition(opts.kmzlog)
  else:
    kmzpos = None

  if opts.triggerlog:
    triggerpos = mav_position.TriggerPosition(opts.triggerlog)
  else:
    triggerpos = None

  # create a simple lens model using the focal length
  C_params = cam_params.CameraParams(lens=opts.lens, sensorwidth=opts.sensorwidth)

  if opts.camera_params:
    C_params.load(opts.camera_params)

  camera_settings = MPSettings(
    [ MPSetting('roll_stabilised', bool, opts.roll_stabilised, 'Roll Stabilised'),
      MPSetting('altitude', int, opts.altitude, 'Altitude', range=(0,10000), increment=1),
      MPSetting('filter_type', str, 'simple', 'Filter Type',
                choice=['simple', 'compactness']),
      MPSetting('fullres', bool, opts.fullres, 'Full Resolution'),
      MPSetting('quality', int, 75, 'Compression Quality', range=(1,100), increment=1),
      MPSetting('thumbsize', int, opts.thumbsize, 'Thumbnail Size', range=(10, 200), increment=1),
      MPSetting('minscore', int, opts.minscore, 'Min Score', range=(0,1000), increment=1, tab='Scoring'),
      MPSetting('brightness', float, 1.0, 'Display Brightness', range=(0.1, 10), increment=0.1,
                digits=2, tab='Display')
      ],
    title='Camera Settings'
    )

  image_settings = MPSettings(
    [ MPSetting('MinRegionArea', float, 0.15, range=(0,100), increment=0.05, digits=2, tab='Image Processing'),
      MPSetting('MaxRegionArea', float, 2.0, range=(0,100), increment=0.1, digits=1),
      MPSetting('MinRegionSize', float, 0.1, range=(0,100), increment=0.05, digits=2),
      MPSetting('MaxRegionSize', float, 2, range=(0,100), increment=0.1, digits=1),
      MPSetting('MaxRarityPct',  float, 0.02, range=(0,100), increment=0.01, digits=2),
      MPSetting('RegionMergeSize', float, 3.0, range=(0,100), increment=0.1, digits=1),
      MPSetting('SaveIntermediate', bool, opts.debug)
      ],
    title='Image Settings')
  
  mosaic = cuav_mosaic.Mosaic(slipmap, C=C_params,
                              camera_settings=camera_settings,
                              image_settings=image_settings,
                              start_menu=True,
                              thumb_size=opts.mosaic_thumbsize)

  joelog = cuav_joe.JoeLog(None)

  if opts.view:
    viewer = mp_image.MPImage(title='Image', can_zoom=True, can_drag=True)

  if camera_settings.filter_type == 'compactness':
    calculate_compactness = True
    print("Using compactness filter")
  else:
    calculate_compactness = False

  for f in files:
      if not mosaic.started():
        print("Waiting for startup")
        while not mosaic.started():
          mosaic.check_events()
          time.sleep(0.01)

      if mpos:
        # get the position by interpolating telemetry data from the MAVLink log file
        # this assumes that the filename contains the timestamp 
        if gamma is not None:
          frame_time = parse_gamma_time(f, gamma)
        else:
          frame_time = cuav_util.parse_frame_time(f)
        frame_time += opts.time_offset
        if camera_settings.roll_stabilised:
          roll = 0
        else:
          roll = None
        try:
          pos = mpos.position(frame_time, roll=roll)
        except Exception:
          print("No position available for %s" % frame_time)
          # skip this frame
          continue
      elif kmzpos is not None:
        pos = kmzpos.position(f)
      elif triggerpos is not None:
        pos = triggerpos.position(f)
      else:
        # get the position using EXIF data
        pos = mav_position.exif_position(f)
        pos.time += opts.time_offset

      # update the plane icon on the map
      if pos is not None:
        slipmap.set_position('plane', (pos.lat, pos.lon), rotation=pos.yaw)
        if camera_settings.altitude > 0:
          pos.altitude = camera_settings.altitude

      # check for any events from the map
      slipmap.check_events()
      mosaic.check_events()

      im_orig = cuav_util.LoadImage(f)
      (w,h) = cuav_util.image_shape(im_orig)

      if not opts.camera_params:
        C_params.set_resolution(w, h)
      
      im_full = im_orig
        
      im_640 = cv.CreateImage((640, 480), 8, 3)
      cv.Resize(im_full, im_640, cv.CV_INTER_NN)
      im_640 = numpy.ascontiguousarray(cv.GetMat(im_640))
      im_full = numpy.ascontiguousarray(cv.GetMat(im_full))

      count = 0
      total_time = 0

      t0=time.time()
      if camera_settings.fullres:
        img_scan = im_full
      else:
        img_scan = im_640

      scan_parms = {}
      for name in image_settings.list():
        scan_parms[name] = image_settings.get(name)
      scan_parms['SaveIntermediate'] = float(scan_parms['SaveIntermediate'])

      if pos is not None:
        (sw,sh) = cuav_util.image_shape(img_scan)
        mpp = cuav_util.meters_per_pixel(pos, C=C_params)
        if mpp is not None:
          scan_parms['MetersPerPixel'] = mpp * (w/float(sw))
        regions = scanner.scan(img_scan, scan_parms)
      else:
        regions = scanner.scan(img_scan)
      regions = cuav_region.RegionsConvert(regions, cuav_util.image_shape(img_scan), cuav_util.image_shape(im_full), calculate_compactness)
      count += 1
      t1=time.time()

      frame_time = pos.time

      regions = cuav_region.filter_regions(im_full, regions, frame_time=frame_time,
                                           min_score=camera_settings.minscore,
                                           filter_type=camera_settings.filter_type)

      scan_count += 1

      mosaic.add_image(pos.time, f, pos)

      if pos and len(regions) > 0:
        altitude = camera_settings.altitude
        if altitude <= 0:
          altitude = None
        joelog.add_regions(frame_time, regions, pos, f, width=w, height=h,
                           altitude=altitude)

      region_count += len(regions)

      if len(regions) > 0:
          composite = cuav_mosaic.CompositeThumbnail(cv.GetImage(cv.fromarray(im_full)), regions)
          thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
          mosaic.add_regions(regions, thumbs, f, pos)

      if opts.view:
        img_view = img_scan
        (wview,hview) = cuav_util.image_shape(img_view)
        mat = cv.fromarray(img_view)
        for r in regions:
          r.draw_rectangle(mat, (255,0,0))
        cv.CvtColor(mat, mat, cv.CV_BGR2RGB)
        viewer.set_image(mat)
        viewer.set_title('Image: ' + os.path.basename(f))

      total_time += (t1-t0)
      if t1 != t0:
          print('%s scan %.1f fps  %u regions [%u/%u]' % (
              os.path.basename(f), count/total_time, region_count, scan_count, num_files))
      #raw_input("hit ENTER when ready")


def parse_args():
  '''parse command line arguments'''
  if 1 == len(sys.argv):
    from MAVProxy.modules.lib.optparse_gui import OptionParser
    file_type='file'
    directory_type='directory'
  else:
    from optparse import OptionParser
    file_type='str'
    directory_type='str'

  parser = OptionParser("geosearch.py [options] <directory>", description='GeoSearch')

  parser.add_option("--directory", default=None, type=directory_type,
                    help="directory containing image files")
  parser.add_option("--mission", default=None, type=file_type, help="mission file to display")
  parser.add_option("--mavlog", default=None, type=file_type, help="MAVLink telemetry log file")
  parser.add_option("--kmzlog", default=None, type=file_type, help="kmz file for image positions")
  parser.add_option("--triggerlog", default=None, type=file_type, help="robota trigger file for image positions")
  parser.add_option("--time-offset", type='int', default=0, help="offset between camera and mavlink log times (seconds)")
  parser.add_option("--view", action='store_true', default=False, help="show images")
  parser.add_option("--lens", default=28.0, type='float', help="lens focal length")
  parser.add_option("--sensorwidth", default=35.0, type='float', help="sensor width")
  parser.add_option("--service", default='MicrosoftSat', help="map service")
  parser.add_option("--camera-params", default=None, type=file_type, help="camera calibration json file from OpenCV")
  parser.add_option("--debug", default=False, action='store_true', help="enable debug info")
  parser.add_option("--fullres", default=False, action='store_true', help="use full camera resolution")
  parser.add_option("--roll-stabilised", default=False, action='store_true', help="assume roll stabilised camera")
  parser.add_option("--altitude", default=0, type='float', help="altitude (0 for auto)")
  parser.add_option("--thumbsize", default=60, type='int', help="thumbnail size")
  parser.add_option("--mosaic-thumbsize", default=35, type='int', help="mosaic thumbnail size")
  parser.add_option("--minscore", default=700, type='int', help="minimum score")
  parser.add_option("--gammalog", default=None, type='str', help="gamma.log from flight")
  return parser.parse_args()

if __name__ == '__main__':
  multiprocessing.freeze_support()
  (opts, args) = parse_args()

  # main program
  if opts.directory is not None:
    process([opts.directory])
  else:
    process(args)
  while True:
    slipmap.check_events()
    mosaic.check_events()
    time.sleep(0.002)
