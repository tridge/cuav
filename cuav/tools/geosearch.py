#!/usr/bin/python

import numpy, os, time, cv2, sys, math, sys, glob, argparse


from cuav.lib import cuav_util, cuav_landingregion
from cuav.image import scanner
from cuav.lib import cuav_mosaic, mav_position, cuav_joe, cuav_region
from cuav.camera import cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.lib import mp_image
from MAVProxy.modules.lib import multiproc
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
  if os.path.isdir(args.directory):
    files.extend(file_list(args.directory, ['jpg', 'jpeg', 'png']))
  else:
    if args.directory.find('*') != -1:
      files.extend(glob.glob(args.directory))
    else:
      files.append(args.directory)
  files.sort()
  num_files = len(files)
  print("num_files=%u" % num_files)
  region_count = 0

  slipmap = mp_slipmap.MPSlipMap(service=args.service, elevation=True, title='Map')
  if args.vehicle_type == "Copter":
      icon = slipmap.icon('redcopter.png')
  else:
      icon = slipmap.icon('redplane.png')
  slipmap.add_object(mp_slipmap.SlipIcon('plane', (0,0), icon, layer=3, rotation=0,
                                         follow=True,
                                         trail=mp_slipmap.SlipTrail()))

  for flag in args.flag:
    a = flag.split(',')
    lat = a[0]
    lon = a[1]
    icon = 'flag.png'
    if len(a) > 2:
      icon = a[2] + '.png'
    icon = slipmap.icon(icon)
    slipmap.add_object(mp_slipmap.SlipIcon('icon - %s' % str(flag), (float(lat),float(lon)), icon, layer=3, rotation=0, follow=False))

  if args.mission:
    from pymavlink import mavwp
    wp = mavwp.MAVWPLoader()
    wp.load(args.mission.name)
    plist = wp.polygon_list()
    if len(plist) > 0:
        for i in range(len(plist)):
          slipmap.add_object(mp_slipmap.SlipPolygon('Mission-%s-%u' % (args.mission.name,i), plist[i], layer='Mission',
                                     linewidth=2, colour=(255,255,255)))

  if args.mavlog:
    mpos = mav_position.MavInterpolator()
    mpos.set_logfile(args.mavlog)
  else:
    mpos = None

  if args.gammalog is not None:
    gamma = parse_gamma_log(args.gammalog)
  else:
    gamma = None

  if args.kmzlog:
    kmzpos = mav_position.KmlPosition(args.kmzlog)
  else:
    kmzpos = None

  if args.triggerlog:
    triggerpos = mav_position.TriggerPosition(args.triggerlog)
  else:
    triggerpos = None

  # create a simple lens model using the focal length

  if args.camera_params:
    C_params = cam_params.CameraParams.fromfile(args.camera_params)
  else:
    im_orig = cv2.imread(files[0])
    (w,h) = cuav_util.image_shape(im_orig)
    C_params = cam_params.CameraParams(lens=args.lens, sensorwidth=args.sensorwidth, xresolution=w, yresolution=h)

  if args.target:
    target = args.target.split(',')
  else:
    target = [0,0,0]

  camera_settings = MPSettings(
    [ MPSetting('roll_stabilised', bool, args.roll_stabilised, 'Roll Stabilised'),
      MPSetting('pitch_stabilised', bool, args.pitch_stabilised, 'Pitch Stabilised'),
      MPSetting('roll_offset', float, args.roll_offset, 'Roll Offset'),
      MPSetting('pitch_offset', float, args.pitch_offset, 'Pitch Offset'),
      MPSetting('altitude', int, args.altitude, 'Altitude', range=(0,10000), increment=1),
      MPSetting('minalt', int, 30, 'MinAltitude', range=(0,10000), increment=1),
      MPSetting('mpp100', float, 0.0977, 'MPPat100m', range=(0,10000), increment=0.001),
      MPSetting('rotate180', bool, False, 'rotate180'),
      MPSetting('showlz', bool, args.showlz, 'Show calculated landing zone from regions'),
      MPSetting('filter_type', str, 'simple', 'Filter Type',
                choice=['simple']),
      MPSetting('target_lattitude', float, float(target[0]), 'target latitude', increment=1.0e-7),
      MPSetting('target_longitude', float, float(target[1]), 'target longitude', increment=1.0e-7),
      MPSetting('target_radius', float, float(target[2]), 'target radius', increment=1),
      MPSetting('quality', int, 75, 'Compression Quality', range=(1,100), increment=1),
      MPSetting('thumbsize', int, args.thumbsize, 'Thumbnail Size', range=(10, 200), increment=1),
      MPSetting('map_thumbsize', int, args.map_thumbsize, 'Map Thumbnail Size', range=(10, 200), increment=1),
      MPSetting('minscore', int, args.minscore, 'Min Score', range=(0,1000), increment=1, tab='Scoring'),
      MPSetting('brightness', float, 1.0, 'Display Brightness', range=(0.1, 10), increment=0.1,
                digits=2, tab='Display'),
      ],
    title='Camera Settings'
    )

  image_settings = MPSettings(
    [ MPSetting('MinRegionArea', float, 0.15, range=(0,100), increment=0.05, digits=2, tab='Image Processing'),
      MPSetting('MaxRegionArea', float, 1.0, range=(0,100), increment=0.1, digits=1),
      MPSetting('MinRegionSize', float, 0.2, range=(0,100), increment=0.05, digits=2),
      MPSetting('MaxRegionSize', float, 1.0, range=(0,100), increment=0.1, digits=1),
      MPSetting('MaxRarityPct',  float, 0.02, range=(0,100), increment=0.01, digits=2),
      MPSetting('RegionMergeSize', float, 1.0, range=(0,100), increment=0.1, digits=1),
      MPSetting('BlueEmphasis', bool, args.blue_emphasis),
      MPSetting('SaveIntermediate', bool, args.debug)
      ],
    title='Image Settings')

  mosaic = cuav_mosaic.Mosaic(slipmap, C=C_params,
                              camera_settings=camera_settings,
                              image_settings=image_settings,
                              start_menu=True,
                              classify=args.categories,
                              thumb_size=args.mosaic_thumbsize,
                              map_thumb_size=args.map_thumbsize)

  joelog = cuav_joe.JoeLog(None)

  lz = cuav_landingregion.LandingZone()

  if args.view:
    viewer = mp_image.MPImage(title='Image', can_zoom=True, can_drag=True)

  start_time = time.time()
  for f in files:
      if not mosaic.started():
        print("Waiting for startup")
        if args.start:
          mosaic.has_started = True
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
        frame_time += args.time_offset
        if camera_settings.roll_stabilised:
          roll = 0
        else:
          roll = None
        if camera_settings.pitch_stabilised:
          pitch = 0
        else:
          pitch = None
        try:
          pos = mpos.position(frame_time, roll=roll, pitch=pitch,
                              roll_offset=camera_settings.roll_offset,
                              pitch_offset=camera_settings.pitch_offset)
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
        pos.time += args.time_offset

      # update the vehicle icon on the map
      if pos is not None:
        slipmap.set_position("plane", (pos.lat, pos.lon), rotation=pos.yaw)
        if camera_settings.altitude > 0:
          pos.altitude = camera_settings.altitude

      # check for any events from the map
      slipmap.check_events()
      mosaic.check_events()

      #im_orig = cuav_util.LoadImage(f, rotate180=camera_settings.rotate180)
      im_orig = cv2.imread(f)

      if im_orig is None:
        continue
      (w,h) = cuav_util.image_shape(im_orig)

      if args.downsample:
        im_full = cv2.resize(im_orig, (0,0), fx=0.5, fy=0.5)
      else:
        im_full = im_orig

      count = 0
      total_time = 0

      t0=time.time()
      img_scan = im_full

      scan_parms = {}
      for name in image_settings.list():
        scan_parms[name] = image_settings.get(name)
      scan_parms['SaveIntermediate'] = float(scan_parms['SaveIntermediate'])
      scan_parms['BlueEmphasis'] = float(scan_parms['BlueEmphasis'])

      if pos is not None:
        (sw,sh) = cuav_util.image_shape(img_scan)
        altitude = pos.altitude
        if altitude < camera_settings.minalt:
          altitude = camera_settings.minalt
        scan_parms['MetersPerPixel'] = camera_settings.mpp100 * altitude / 100.0

        regions = scanner.scan(img_scan, scan_parms)
      else:
        regions = scanner.scan(img_scan)
      regions = cuav_region.RegionsConvert(regions, cuav_util.image_shape(img_scan), cuav_util.image_shape(im_full))
      count += 1
      t1=time.time()

      frame_time = pos.time

      if pos:
        for r in regions:
          r.latlon = cuav_util.gps_position_from_image_region(r, pos, w, h, altitude=altitude, C=C_params)

        if camera_settings.target_radius > 0 and pos is not None:
          regions = cuav_region.filter_radius(regions, (camera_settings.target_lattitude,
                                                        camera_settings.target_longitude),
                                              camera_settings.target_radius)

      regions = cuav_region.filter_regions(im_full, regions, min_score=camera_settings.minscore,
                                           filter_type=camera_settings.filter_type)

      scan_count += 1

      if pos and len(regions) > 0:
        joelog.add_regions(frame_time, regions, pos, f)

      mosaic.add_image(pos.time, f, pos)

      if camera_settings.showlz:
          for r in regions:
            if r.score >= camera_settings.minscore and pos is not None:
                lz.checkaddregion(r, pos)
          #if lz.landingzoneconfidence and lz.landingzoneconfidence < 20:
          #  print("Refining LZ")
          lzresult = lz.calclandingzone()
          if lzresult is not None:
            slipmap.add_object(mp_slipmap.SlipCircle('LZ', 'LZ', lzresult.latlon, lzresult.maxrange,
                                     linewidth=3, color=(0,255,0)))
            slipmap.add_object(mp_slipmap.SlipCircle('LZMid', 'LZMid', lzresult.latlon, 2.0,
                                     linewidth=3, color=(0,255,0)))

      region_count += len(regions)

      if len(regions) > 0:
          composite = cuav_region.CompositeThumbnail(im_full, regions)
          thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
          thumbsRGB = []

          #colour space conversion
          for thumb in thumbs:
              thumbsRGB.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
          mosaic.add_regions(regions, thumbsRGB, f, pos)

      if args.view:
        img_view = img_scan
        (wview,hview) = cuav_util.image_shape(img_view)
        for r in regions:
          r.draw_rectangle(img_view, (255,0,0))
        #cv.CvtColor(mat, mat, cv.CV_BGR2RGB)
        img_view = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
        viewer.set_image(img_view)
        viewer.set_title('Image: ' + os.path.basename(f))

      total_time += (t1-t0)
      if t1 != t0:
          print('%s scan %.1f fps  %u regions [%u/%u]' % (
              os.path.basename(f), count/total_time, region_count, scan_count, num_files))
      #raw_input("hit ENTER when ready")

  print("All images processed (%u seconds)" % (time.time() - start_time))
  while True:
      # check for any events from the map
      slipmap.check_events()
      mosaic.check_events()
      time.sleep(0.2)

def parse_args():
    '''parse command line arguments'''
    parser = argparse.ArgumentParser(description='Search images for Joe')
    parser.add_argument("--vehicle-type", default="Plane", help="vehicle type", choices=('Plane','Copter'))
    parser.add_argument("directory", default=None, help="directory containing image files")
    parser.add_argument("--mission", default=None, type=str, help="mission file to display")
    parser.add_argument("--mavlog", default=None, type=str, help="MAVLink telemetry log file")
    parser.add_argument("--kmzlog", default=None, type=str, help="kmz file for image positions")
    parser.add_argument("--triggerlog", default=None, type=str, help="robota trigger file for image positions")
    parser.add_argument("--time-offset", type=float, default=0, help="offset between camera and mavlink log times (seconds)")
    parser.add_argument("--view", action='store_true', default=False, help="show images")
    parser.add_argument("--lens", default=28.0, type=float, help="lens focal length")
    parser.add_argument("--sensorwidth", default=35.0, type=float, help="sensor width")
    parser.add_argument("--service", default='MicrosoftSat', choices=('GoogleSat', 'MicrosoftSat', 'OviSat', 'OpenStreetMap', 'MicrosoftHyb', 'OviHybrid', 'GoogleMap'), help="map service")
    parser.add_argument("--camera-params", default=None, type=str, help="camera calibration json file from OpenCV")
    parser.add_argument("--debug", default=False, action='store_true', help="enable debug info")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="assume roll stabilised camera")
    parser.add_argument("--pitch-stabilised", default=False, action='store_true', help="assume pitch stabilised camera")
    parser.add_argument("--pitch-offset", default=0, type=float, help="pitch offset from autopilot pitch")
    parser.add_argument("--roll-offset", default=0, type=float, help="roll offset from autopilot pitch")
    parser.add_argument("--altitude", default=0, type=float, help="altitude (0 for auto)")
    parser.add_argument("--thumbsize", default=60, type=int, help="thumbnail size")
    parser.add_argument("--map-thumbsize", default=60, type=int, help="map thumbnail size")
    parser.add_argument("--mosaic-thumbsize", default=35, type=int, help="mosaic thumbnail size")
    parser.add_argument("--minscore", default=100, type=int, help="minimum score")
    parser.add_argument("--gammalog", default=None, type=str, help="gamma.log from flight")
    parser.add_argument("--target", default=None, type=str, help="lat,lon,radius target")
    parser.add_argument("--categories", default=None, type=str, help="xml file containing categories for classification")
    if 1 != len(sys.argv):
        parser.add_argument("--flag", default=[], type=str, action='append', help="flag positions"),
    parser.add_argument("--blue-emphasis", default=False, action='store_true', help="enable blue emphasis in scanner")
    parser.add_argument("--start", default=False, action='store_true', help="start straight away")
    parser.add_argument("--downsample", default=False, action='store_true', help="downsample image before scanning")
    parser.add_argument("--showlz", default=False, action='store_true', help="Show calculated landing zone from regions")
    return parser.parse_args()



if __name__ == '__main__':
    multiproc.freeze_support()
    args = parse_args()

    # main program
    process(args)
    while True:
        slipmap.check_events()
        mosaic.check_events()
        time.sleep(0.002)
