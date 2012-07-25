#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util, time, math, functools, cuav_region

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy', 'modules', 'lib'))
import scanner
from cam_params import CameraParams

class MosaicRegion:
    def __init__(self, region, filename, pos, thumbnail, latlon=(None,None)):
        # self.region is a (minx,miny,maxy,maxy) rectange in image coordinates
        self.region = region
        self.filename = filename
        self.pos = pos
        self.thumbnail = thumbnail
        self.latlon = latlon

    def __str__(self):
        position_string = ''
        if self.latlon != (None,None):
            position_string += '%s ' % str(self.latlon)
        if self.pos != None:
            position_string += ' %s %s' % (str(self.pos), time.asctime(time.localtime(self.pos.time)))
        return '%s %s' % (position_string, self.filename)


class MosaicImage:
    def __init__(self, filename, pos, boundary, center):
        self.filename = filename
        self.pos = pos
        self.boundary = boundary
        self.center = center

    def __str__(self):
        return '%s %s' % (self.filename, str(self.pos))

class DisplayedImage:
    def __init__(self, filename, pos, img):
        self.filename = filename
        self.pos = pos
        self.img = img

    def __str__(self):
        return '%s %s' % (self.filename, str(self.pos))

def CompositeThumbnail(img, regions, thumb_size=100, quality=75, xsize=640):
    '''extract a composite thumbnail for the regions of an image

    The composite will consist of N thumbnails side by side

    return it as a compressed jpeg string
    '''
    composite = cv.CreateImage((thumb_size*len(regions), thumb_size),8,3)
    for i in range(len(regions)):
        (x1,y1,x2,y2) = regions[i].tuple()
        midx = (x1+x2)/2
        midy = (y1+y2)/2

        if cuav_util.image_width(img) == 1280 and xsize==640:
            # the regions are from a 640x480 image. If we are extracting
            # from a 1280x960, then move the central pixel
            midx *= 2
            midy *= 2

        x1 = midx - thumb_size/2
        y1 = midy - thumb_size/2
        thumb = cuav_util.SubImage(img, (x1, y1, thumb_size, thumb_size))
        cv.SetImageROI(composite, (thumb_size*i, 0, thumb_size, thumb_size))
        cv.Copy(thumb, composite)
        cv.ResetImageROI(composite)
    return scanner.jpeg_compress(numpy.ascontiguousarray(cv.GetMat(composite)), quality)

def ExtractThumbs(img, count):
    '''extract thumbnails from a composite thumbnail image'''
    thumb_size = cuav_util.image_width(img) / count
    thumbs = []
    for i in range(count):
        thumb = cuav_util.SubImage(img, (i*thumb_size, 0, thumb_size, thumb_size))
        thumbs.append(thumb)
    return thumbs

class Mosaic():
    '''keep a mosaic of found regions'''
    def __init__(self, slipmap,
                 grid_width=20, grid_height=20, thumb_size=30, C=CameraParams()):
        self.thumb_size = thumb_size
        self.width = grid_width * thumb_size
        self.height = grid_height * thumb_size
        self.mosaic = cv.CreateImage((self.height,self.width),8,3)
        self.display_regions = grid_width*grid_height
        self.regions = []
        self.images = []
        self.full_res = False
        self.boundary = []
        self.displayed_image = None
        self.last_click_position = None
        self.c_params = C
        import mp_image
        self.image_mosaic = mp_image.MPImage(title='Mosaic')
        self.slipmap = slipmap

        self.slipmap.add_callback(functools.partial(self.map_callback))

    def map_callback(self, event):
        '''called when an event happens on the slipmap'''
        import mp_slipmap
        if not isinstance(event, mp_slipmap.SlipMouseEvent):
            return
        if len(event.selected) == 0:
            # no objects were selected
            return
        # use just the first elected object, which is the one
        # closest to the mouse position
        key = str(event.selected[0].objkey)
        if not key.startswith("region "):
            return
        r = key.split()
        ridx = int(r[1])
        print("Selected %s ridx=%u" % (key, ridx))
        if ridx < 0 or ridx >= len(self.regions):
            print("Invalid region %u selected" % ridx)
            return

        region = self.regions[ridx]
        thumbnail = cv.CloneImage(region.thumbnail)
        # slipmap wants it as RGB
        import mp_slipmap
        cv.CvtColor(thumbnail, thumbnail, cv.CV_BGR2RGB)
        thumbnail_saturated = cuav_util.SaturateImage(thumbnail)
        self.slipmap.add_object(mp_slipmap.SlipInfoImage('region saturated', thumbnail_saturated))
        self.slipmap.add_object(mp_slipmap.SlipInfoImage('region detail', thumbnail))
        region_text = "Selected region %u score=%u\n%s\n%s" % (ridx, region.region.score,
                                                               str(region.latlon), os.path.basename(region.filename))
        self.slipmap.add_object(mp_slipmap.SlipInfoText('region detail text', region_text))
            

    def set_boundary(self, boundary):
        '''set a polygon search boundary'''
        import mp_slipmap
        if not cuav_util.polygon_complete(boundary):
            raise RuntimeError('invalid boundary passed to mosaic')
        self.boundary = boundary[:]
        self.slipmap.add_object(mp_slipmap.SlipPolygon('boundary', self.boundary, layer=1, linewidth=2, colour=(0,255,0)))


    def display_region_image(self, region):
        '''display the thumbnail associated with a region'''
        print('-> %s' % str(region))
        img = cv.fromarray(region.thumbnail)

    def mouse_event(self, event, x, y, flags, data):
        '''called on mouse events'''
        if flags & cv.CV_EVENT_FLAG_RBUTTON:
            self.full_res = not self.full_res
            print("full_res: %s" % self.full_res)
        if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
            return
        # work out which region they want, taking into account wrap
        idx = (x/self.thumb_size) + (self.width/self.thumb_size)*(y/self.thumb_size)
        if idx >= len(self.regions):
            return
        first = (len(self.regions)/self.display_regions) * self.display_regions
        idx += first
        if idx >= len(self.regions):
            idx -= self.display_regions
        region = self.regions[idx]

        self.display_region_image(region)

    def mouse_event_image(self, event, x, y, flags, data):
        '''called on mouse events on a displayed image'''
        if not (flags & cv.CV_EVENT_FLAG_LBUTTON):
            return
        img = self.displayed_image
        if img is None or img.pos is None:
            return
        pos = img.pos
        (w,h) = cuav_util.image_shape(img.img)
        latlon = cuav_util.pixel_coordinates(x, y, pos.lat, pos.lon, pos.altitude,
                                             pos.pitch, pos.roll, pos.yaw,
                                             xresolution=w, yresolution=h,
                                             C=self.c_params)
        if latlon is None:
            print("Unable to find pixel coordinates")
            return
        (lat,lon) = latlon
        print("=> %s %f %f %s" % (img.filename, lat, lon, str(pos)))
        if self.last_click_position:
            (last_lat, last_lon) = self.last_click_position
            print("distance from last click: %.1f m" % (cuav_util.gps_distance(lat, lon, last_lat, last_lon)))
        self.last_click_position = (lat,lon)


    def image_boundary(self, img, pos):
        '''return a set of 4 (lat,lon) coordinates for the 4 corners
        of an image in the order
        (left,top), (right,top), (right,bottom), (left,bottom)

        Note that one or more of the corners may return as None if
        the corner is not on the ground (it points at the sky)
        '''
        (w,h) = cuav_util.image_shape(img)
        # scale to sensor dimensions
        scale_x = float(self.c_params.xresolution)/float(w)
        scale_y = float(self.c_params.yresolution)/float(h)
        latlon = []
        for (x,y) in [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]:
            x *=scale_x
	    y *=scale_y
            latlon.append(cuav_util.pixel_coordinates(x, y, pos.lat, pos.lon, pos.altitude,
                                                      pos.pitch, pos.roll, pos.yaw,
                                                      C=self.c_params))
        # make it a complete polygon by appending the first point
        latlon.append(latlon[0])
        return latlon

    def image_center(self, img, pos):
        '''return a (lat,lon) for the center of the image
        '''
        (w,h) = cuav_util.image_shape(img)
        # scale to sensor dimensions
        scale_x = float(self.c_params.xresolution)/float(w)
        scale_y = float(self.c_params.yresolution)/float(h)
        x = scale_x*w
        y = scale_y*h
        return cuav_util.pixel_coordinates(x, y, pos.lat, pos.lon, pos.altitude,
                                           pos.pitch, pos.roll, pos.yaw,
                                           C=self.c_params)

    def image_area(self, corners):
        '''return approximage area of an image delimited by the 4 corners
        use the min and max coordinates, thus giving a overestimate
        '''
        minx = corners[0][0]
        maxx = corners[0][0]
        miny = corners[0][1]
        maxy = corners[0][1]
        for (x,y) in corners:
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
        return (maxy-miny)*(maxx-minx)


    def add_regions(self, regions, thumbs, filename, pos=None):
        '''add some regions'''
        for i in range(len(regions)):
            r = regions[i]
            (x1,y1,x2,y2) = r.tuple()

            (lat, lon) = r.latlon

            if self.boundary and (lat,lon) == (None,None):
                # its pointing into the sky
                continue
            if self.boundary:
                if cuav_util.polygon_outside((lat, lon), self.boundary):
                    # this region is outside the search boundary
                    continue

            # the thumbnail we have been given will be bigger than the size we want to
            # display on the mosaic. Extract the middle of it for display
            full_thumb = thumbs[i]
            tsize = cuav_util.image_width(full_thumb)
            thumb = cuav_util.SubImage(full_thumb, ((tsize-self.thumb_size)//2,
                                                    (tsize-self.thumb_size)//2,
                                                    self.thumb_size,
                                                    self.thumb_size))

            idx = len(self.regions) % self.display_regions

            dest_x = (idx * self.thumb_size) % self.width
            dest_y = ((idx * self.thumb_size) / self.width) * self.thumb_size

            # overlay thumbnail on mosaic
            cuav_util.OverlayImage(self.mosaic, thumb, dest_x, dest_y)

            # use the index into self.regions[] as the key for thumbnails
            # displayed on the map
            ridx = len(self.regions)

            if (lat,lon) != (None,None):
                import mp_slipmap
                self.slipmap.add_object(mp_slipmap.SlipThumbnail("region %u" % ridx, (lat,lon),
                                                                 img=thumb,
                                                                 layer=2, border_width=1, border_colour=(255,0,0)))

            self.regions.append(MosaicRegion(r, filename, pos, thumbs[i], latlon=(lat, lon)))

        self.image_mosaic.set_image(self.mosaic, bgr=True)
#      cv.SetMouseCallback('Mosaic', self.mouse_event, self)
