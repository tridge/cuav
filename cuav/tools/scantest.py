#!/usr/bin/python

import numpy, os, time, cv2, sys, sys, glob
from argparse import ArgumentParser

from cuav.image import scanner
from cuav.lib import cuav_util, cuav_mosaic, mav_position, cuav_joe, cuav_region
from cuav.camera import cam_params
from MAVProxy.modules.mavproxy_map import mp_slipmap
from MAVProxy.modules.lib import mp_image


slipmap = None
mosaic = None

def scan_image_directory(dirname):
    '''scan a image directory, extracting frame_time and filename
    as a list of tuples'''
    ret = []
    if os.path.isfile(dirname):
        return [dirname]
    types = ('*.png', '*.jpeg', '*.jpg')
    for tp in types:
        for f in glob.iglob(os.path.join(dirname, tp)):
            ret.append(f)
    return ret

def process(args):
    '''process a set of files'''

    global slipmap, mosaic
    scan_count = 0
    files = scan_image_directory(args.imagedir)
    files.sort()
    num_files = len(files)
    print("num_files=%u" % num_files)
    region_count = 0
    joes = []

    if args.mavlog:
        mpos = mav_position.MavInterpolator(gps_lag=args.gps_lag)
        mpos.set_logfile(args.mavlog)
    else:
        mpos = None

    if args.boundary:
        boundary = cuav_util.polygon_load(args.boundary)
    else:
        boundary = None

    if args.mosaic:
        slipmap = mp_slipmap.MPSlipMap(service='GoogleSat', elevation=True, title='Map')
        icon = slipmap.icon('redplane.png')
        slipmap.add_object(mp_slipmap.SlipIcon('plane', (0,0), icon, layer=3, rotation=0,
                                         follow=True,
                                         trail=mp_slipmap.SlipTrail()))
        if args.camera_params:
            C_params = cam_params.CameraParams.fromfile(args.camera_params.name)
        else:
            im_orig = cv2.imread(files[0])
            (w,h) = cuav_util.image_shape(im_orig)
            C_params = cam_params.CameraParams(lens=args.lens, sensorwidth=args.sensorwidth, xresolution=w, yresolution=h)
        mosaic = cuav_mosaic.Mosaic(slipmap, C=C_params)
        if boundary is not None:
            mosaic.set_boundary(boundary)

    if args.joe:
        joes = cuav_util.polygon_load(args.joe)
        if boundary:
            for i in range(len(joes)):
                joe = joes[i]
                if cuav_util.polygon_outside(joe, boundary):
                    print("Error: joe outside boundary", joe)
                    return
                icon = slipmap.icon('flag.png')
                slipmap.add_object(mp_slipmap.SlipIcon('joe%u' % i, (joe[0],joe[1]), icon, layer=4))

    joelog = cuav_joe.JoeLog('joe.log')      

    if args.view:
        viewer = mp_image.MPImage(title='Image')

    frame_time = 0

    scan_parms = {
        'MinRegionArea' : args.min_region_area,
        'MaxRegionArea' : args.max_region_area,
        'MinRegionSize' : args.min_region_size,
        'MaxRegionSize' : args.max_region_size,
        'MaxRarityPct'  : args.max_rarity_pct,
        'RegionMergeSize' : args.region_merge,
        'SaveIntermediate' : float(0),
        #'SaveIntermediate' : float(args.debug),
        'MetersPerPixel' : args.meters_per_pixel100 * args.altitude / 100.0
    }

    filenum = 0
        
    for f in files:
        filenum += 1
        if mpos:
            frame_time = cuav_util.parse_frame_time(f)
            try:
                if args.roll_stabilised:
                    roll = 0
                else:
                    roll = None
                pos = mpos.position(frame_time, args.max_deltat,roll=roll)
                slipmap.set_position('plane', (pos.lat, pos.lon), rotation=pos.yaw)
            except mav_position.MavInterpolatorException as e:
                print(e)
                pos = None
        else:
              pos = None

        # check for any events from the map
        if args.mosaic:
            slipmap.check_events()
            mosaic.check_events()

        im_orig = cv2.imread(f)
        (w,h) = cuav_util.image_shape(im_orig)
        im_full = im_orig
        im_half = cv2.resize(im_orig, (0,0), fx=0.5, fy=0.5)
        im_half = numpy.ascontiguousarray(im_half)
        im_full = numpy.ascontiguousarray(im_full)

        count = 0
        total_time = 0
        if args.fullres:
            img_scan = im_full
        else:
            img_scan = im_half

        t0=time.time()
        for i in range(args.repeat):
            regions = scanner.scan(img_scan, scan_parms)
            regions = cuav_region.RegionsConvert(regions, cuav_util.image_shape(img_scan), cuav_util.image_shape(im_full))
            count += 1
        t1=time.time()

        if args.filter:
            regions = cuav_region.filter_regions(im_full, regions, min_score=args.minscore,
                                           filter_type=args.filter_type)

        if len(regions) > 0 and args.debug:
            composite = cuav_region.CompositeThumbnail(im_full, regions, thumb_size=args.thumb_size)
            thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
            thumb_num = 0
            for thumb in thumbs:
                print("thumb %u score %f" % (thumb_num, regions[thumb_num].score))
                cv2.imwrite('%u_thumb%u.jpg' % (filenum,thumb_num), thumb)
                thumb_num += 1
            
        scan_count += 1

        # optionally link all the images with joe into a separate directory
        # for faster re-running of the test with just joe images
        if pos and args.linkjoe and len(regions) > 0:
            cuav_util.mkdir_p(args.linkjoe)
            if not cuav_util.polygon_outside((pos.lat, pos.lon), boundary):
                joepath = os.path.join(args.linkjoe, os.path.basename(f))
                if os.path.exists(joepath):
                    os.unlink(joepath)
                os.symlink(f, joepath)

        if pos and len(regions) > 0:
            joelog.add_regions(frame_time, regions, pos, f, width=w, height=h, altitude=args.altitude, C=C_params)

        if boundary:
            regions = cuav_region.filter_boundary(regions, boundary, pos)

        region_count += len(regions)

        if args.mosaic and len(regions) > 0 and pos:
            composite = cuav_region.CompositeThumbnail(im_full, regions)
            thumbs = cuav_mosaic.ExtractThumbs(composite, len(regions))
            mosaic.add_regions(regions, thumbs, f, pos)

        if args.view:
            if args.fullres:
                img_view = im_full
            else:
                img_view = img_scan
            #mat = cv.fromarray(img_view)
            for r in regions:
                r.draw_rectangle(img_view, colour=(255,0,0), linewidth=min(max(w/600,1),3), offset=max(w/200,1))
            img_view = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
            viewer.set_image(img_view)

        total_time += (t1-t0)
        if t1 != t0:
            print('%s scan %.1f fps  %u regions [%u/%u]' % (
                f, count/total_time, region_count, scan_count, num_files))


# main program
if __name__ == '__main__':
    parser = ArgumentParser(description="Scanner test")
    parser.add_argument("imagedir", default=None, help='image directory')
    parser.add_argument("--repeat", type=int, default=1, help="scan repeat count")
    parser.add_argument("--view", action='store_true', default=False, help="show images")
    parser.add_argument("--fullres", action='store_true', default=True, help="show full resolution")
    parser.add_argument("--gamma", type=int, default=0, help="gamma for 16 -> 8 conversion")
    parser.add_argument("--mosaic", action='store_true', default=False, help="build a mosaic of regions")
    parser.add_argument("--mavlog", default=None, help="flight log for geo-referencing")
    parser.add_argument("--boundary", default=None, help="search boundary file")
    parser.add_argument("--max-deltat", default=0.0, type=float, help="max deltat for interpolation")
    parser.add_argument("--max-attitude", default=45, type=float, help="max attitude geo-referencing")
    parser.add_argument("--joe", default=None, help="file containing list of joe positions")
    parser.add_argument("--linkjoe", default=None, help="link joe images to this directory")
    parser.add_argument("--lens", default=4.0, type=float, help="lens focal length")
    parser.add_argument("--roll-stabilised", default=False, action='store_true', help="roll is stabilised")
    parser.add_argument("--gps-lag", default=0.0, type=float, help="GPS lag in seconds")
    parser.add_argument("--filter", default=False, action='store_true', help="filter using HSV")
    parser.add_argument("--minscore", default=3, type=int, help="minimum score")
    parser.add_argument("--altitude", type=int, default=None, help="camera assumed altitude")
    parser.add_argument("--filter-type", default='simple', choices=['simple'], help="object filter type")
    parser.add_argument("--min-region-area", default=0.15, type=float, help="minimum region area (m^2)")
    parser.add_argument("--max-region-area", default=1.0, type=float, help="maximum region area (m^2)")
    parser.add_argument("--min-region-size", default=0.2, type=float, help="minimum region size (m)")
    parser.add_argument("--max-region-size", default=1.0, type=float, help="maximum region size (m)")
    parser.add_argument("--region-merge", default=1.0, type=float, help="region merge size (m)")
    parser.add_argument("--max-rarity-pct", default=0.02, type=float, help="maximum percentage rarity (percent)")
    parser.add_argument("--meters-per-pixel100", default=0.0977, type=float, help="meters per pixel at 100m")
    parser.add_argument("--sensorwidth", default=35.0, type=float, help="sensor width")
    parser.add_argument("--camera-params", default=None, type=file, help="camera calibration json file from OpenCV")
    parser.add_argument("--debug", default=False, action='store_true', help="enable debug info")
    parser.add_argument("--thumb-size", default=100, type=int, help="thumbnail size")

    args = parser.parse_args()
    process(args)
    if not args.mosaic and not args.view:
        sys.exit(0)
    while True:
        if args.mosaic:
            slipmap.check_events()
            mosaic.check_events()
        time.sleep(0.01)

