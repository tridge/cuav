#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util, time, math, functools, cuav_region

from MAVProxy.modules.mavproxy_map import mp_image, mp_slipmap
from cuav.image import scanner
from cuav.camera.cam_params import CameraParams

class MosaicRegion:
    def __init__(self, ridx, region, filename, pos, full_thumbnail, small_thumbnail, latlon=(None,None)):
        # self.region is a (minx,miny,maxy,maxy) rectange in image coordinates
        self.region = region
        self.filename = filename
        self.pos = pos
        self.full_thumbnail = full_thumbnail
        self.small_thumbnail = small_thumbnail
        self.latlon = latlon
        self.ridx = ridx
        self.score = region.score

    def __str__(self):
        position_string = ''
        if self.latlon != (None,None):
            position_string += '%s ' % str(self.latlon)
        if self.pos != None:
            position_string += ' %s %s' % (str(self.pos), time.asctime(time.localtime(self.pos.time)))
        return '%s %s' % (position_string, self.filename)


class MosaicImage:
    def __init__(self, frame_time, filename, pos):
        self.frame_time = frame_time
        self.filename = filename
        self.pos = pos

    def __str__(self):
        return '%s %s' % (self.filename, str(self.pos))

def CompositeThumbnail(img, regions, thumb_size=100):
    '''extract a composite thumbnail for the regions of an image

    The composite will consist of N thumbnails side by side
    '''
    composite = cv.CreateImage((thumb_size*len(regions), thumb_size),8,3)
    for i in range(len(regions)):
        (x1,y1,x2,y2) = regions[i].tuple()
        midx = (x1+x2)/2
        midy = (y1+y2)/2

        x1 = midx - thumb_size/2
        y1 = midy - thumb_size/2
        thumb = cuav_util.SubImage(img, (x1, y1, thumb_size, thumb_size))
        cv.SetImageROI(composite, (thumb_size*i, 0, thumb_size, thumb_size))
        cv.Copy(thumb, composite)
        cv.ResetImageROI(composite)
    return composite

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
        cuav_util.zero_image(self.mosaic)
        self.display_regions = grid_width*grid_height
        self.regions = []
        self.regions_sorted = []
        self.page = 0
        self.images = []
        self.current_view = 0
        self.full_res = False
        self.boundary = []
        self.displayed_image = None
        self.last_click_position = None
        self.c_params = C
        import wx
        self.image_mosaic = mp_image.MPImage(title='Mosaic', events=[wx.EVT_MOUSE_EVENTS, wx.EVT_KEY_DOWN])
        self.slipmap = slipmap
        self.selected_region = 0

        self.view_image = None
        self.brightness = 1

        self.slipmap.add_callback(functools.partial(self.map_callback))

    def set_brightness(self, b):
        '''set mosaic brightness'''
        self.brightness = b

    def show_region(self, ridx):
        '''display a region on the map'''
        region = self.regions[ridx]
        thumbnail = cv.CloneImage(region.full_thumbnail)
        # slipmap wants it as RGB
        cv.CvtColor(thumbnail, thumbnail, cv.CV_BGR2RGB)
        thumbnail_saturated = cuav_util.SaturateImage(thumbnail)
        self.slipmap.add_object(mp_slipmap.SlipInfoImage('region saturated', thumbnail_saturated))
        self.slipmap.add_object(mp_slipmap.SlipInfoImage('region detail', thumbnail))
        self.selected_region = ridx
        if region.score is None:
            region.score = 0
        region_text = "Selected region %u score=%u %s\n%s\n%s" % (ridx, region.score,
                                                                  region.region.center(),
                                                                  str(region.latlon), os.path.basename(region.filename))
        self.slipmap.add_object(mp_slipmap.SlipInfoText('region detail text', region_text))

    def show_closest(self, latlon):
        '''show closest camera image'''
        (lat, lon) = latlon
        closest = -1
        closest_distance = -1
        for idx in range(len(self.images)):
            pos = self.images[idx].pos
            if pos is not None:
                distance = cuav_util.gps_distance(lat, lon, pos.lat, pos.lon)
                if closest == -1 or distance < closest_distance:
                    closest_distance = distance
                    closest = idx
        if closest == -1:
            return
        self.current_view = closest
        image = self.images[closest]
        img = cv.LoadImage(image.filename)
        if self.view_image is None or not self.view_image.is_alive():
            import wx
            self.view_image = mp_image.MPImage(title='View', events=[wx.EVT_MOUSE_EVENTS, wx.EVT_KEY_DOWN])
        im_1280 = cv.CreateImage((1280, 960), 8, 3)
        cv.Resize(img, im_1280, cv.CV_INTER_NN)
        self.view_image.set_image(im_1280, bgr=True)

    def map_callback(self, event):
        '''called when an event happens on the slipmap'''
        if not isinstance(event, mp_slipmap.SlipMouseEvent):
            return
        if event.event.m_middleDown:
            # show closest image from history
            self.show_closest(event.latlon)
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
        self.show_region(ridx)
            

    def set_boundary(self, boundary):
        '''set a polygon search boundary'''
        if not cuav_util.polygon_complete(boundary):
            raise RuntimeError('invalid boundary passed to mosaic')
        self.boundary = boundary[:]
        self.slipmap.add_object(mp_slipmap.SlipPolygon('boundary', self.boundary, layer=1, linewidth=2, colour=(0,0,255)))


    def mouse_event(self, event):
        '''called on mouse events on the mosaic'''
        # work out which region they want, taking into account wrap
        x = event.X
        y = event.Y
        page_idx = (x/self.thumb_size) + (self.width/self.thumb_size)*(y/self.thumb_size)
        ridx = page_idx + self.page * self.display_regions
        if ridx >= len(self.regions):
            return
        region = self.regions_sorted[ridx]
        self.show_region(region.ridx)
        if region.latlon != (None,None):
            self.slipmap.add_object(mp_slipmap.SlipCenter(region.latlon))

    def mouse_event_view(self, event):
        '''called on mouse events in View window'''
        x = event.X
        y = event.Y
        if self.current_view >= len(self.images):
            return
        image = self.images[self.current_view]
        latlon = cuav_util.gps_position_from_xy(x, y, image.pos, C=self.c_params)
        print("-> %s %s" % (latlon, image.filename))

    def key_event(self, event):
        '''called on key events'''
        last_page = self.page
        if event.KeyCode == ord('S'):
            self.regions_sorted.sort(key = lambda r : r.score, reverse=True)
            self.redisplay_mosaic()
        if event.KeyCode == ord('H'):
            self.regions_sorted[0].score = 0
            self.regions_sorted = self.regions_sorted[1:] + self.regions_sorted[:1]
            self.redisplay_mosaic()
        if event.KeyCode == ord('M'):
            self.regions[self.selected_region].score *= 2
        if event.KeyCode == ord('R'):
            for r in self.regions_sorted:
                r.score = r.region.score
            self.redisplay_mosaic()
        if event.KeyCode == ord('T'):
            self.regions_sorted.sort(key = lambda r : r.ridx)
            self.redisplay_mosaic()
        if event.KeyCode == ord('N'):
            self.page += 1
        if event.KeyCode == ord('P'):
            self.page -= 1
        if self.page < 0:
            self.page = 0
        if self.page > len(self.regions) / self.display_regions:
            self.page = len(self.regions) / self.display_regions
        if last_page != self.page:
            print("Page %u" % self.page)
            self.redisplay_mosaic()

    def display_mosaic_region(self, ridx):
        '''display a thumbnail on the mosaic'''
        region = self.regions_sorted[ridx]
        page_idx = ridx - self.page * self.display_regions
        if page_idx < 0 or page_idx >= self.display_regions:
            # its not on this page
            return
        dest_x = (page_idx * self.thumb_size) % self.width
        dest_y = ((page_idx * self.thumb_size) / self.width) * self.thumb_size

        # overlay thumbnail on mosaic
        cuav_util.OverlayImage(self.mosaic, region.small_thumbnail, dest_x, dest_y)

    def redisplay_mosaic(self):
        '''re-display whole mosaic page'''
        self.mosaic = cv.CreateImage((self.height,self.width),8,3)
        cuav_util.zero_image(self.mosaic)
        for ridx in range(len(self.regions)):
            self.display_mosaic_region(ridx)
        if self.brightness != 1.0:
            cv.ConvertScale(self.mosaic, self.mosaic, scale=self.brightness)
        self.image_mosaic.set_image(self.mosaic, bgr=True)

    def add_regions(self, regions, thumbs, filename, pos=None):
        '''add some regions'''
        for i in range(len(regions)):
            r = regions[i]
            (x1,y1,x2,y2) = r.tuple()

            latlon = r.latlon
            if latlon is None:
                latlon = (None,None)

            (lat, lon) = latlon

            if self.boundary:
                if (lat, lon) == (None,None):
                    # its pointing into the sky
                    continue
#                if cuav_util.polygon_outside((lat,lon), self.boundary):
#                    # this region is outside the search boundary
#                    continue

            # the thumbnail we have been given will be bigger than the size we want to
            # display on the mosaic. Extract the middle of it for display
            full_thumb = thumbs[i]
            tsize = cuav_util.image_width(full_thumb)
            thumb = cuav_util.SubImage(full_thumb, ((tsize-self.thumb_size)//2,
                                                    (tsize-self.thumb_size)//2,
                                                    self.thumb_size,
                                                    self.thumb_size))

            ridx = len(self.regions)
            self.regions.append(MosaicRegion(ridx, r, filename, pos, thumbs[i], thumb, latlon=(lat,lon)))
            self.regions_sorted.append(self.regions[-1])

            self.display_mosaic_region(ridx)

            if (lat,lon) != (None,None):
                self.slipmap.add_object(mp_slipmap.SlipThumbnail("region %u" % ridx, (lat,lon),
                                                                 img=thumb,
                                                                 layer=2, border_width=1, border_colour=(255,0,0)))

        self.image_mosaic.set_image(self.mosaic, bgr=True)

    def add_image(self, frame_time, filename, pos):
        '''add a camera image'''
        self.images.append(MosaicImage(frame_time, filename, pos))

    def check_events(self):
        '''check for mouse/keyboard events'''
        if self.image_mosaic.is_alive():
            for event in self.image_mosaic.events():
                if event.ClassName == 'wxMouseEvent':
                    self.mouse_event(event)
                if event.ClassName == 'wxKeyEvent':
                    self.key_event(event)
        if self.view_image and self.view_image.is_alive():
            for event in self.view_image.events():
                if event.ClassName == 'wxMouseEvent':
                    self.mouse_event_view(event)
            
        
