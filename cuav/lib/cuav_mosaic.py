#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util, time, math, functools, cuav_region

from MAVProxy.modules.lib import mp_image
from MAVProxy.modules.mavproxy_map import mp_slipmap
from cuav.image import scanner
from cuav.camera.cam_params import CameraParams
from MAVProxy.modules.lib.mp_menu import *
from MAVProxy.modules.lib.wxsettings import WXSettings

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

    def tag_image_available(self, color=(0,255,255)):
        '''tag the small thumbnail image with a marker making it clear the
        full image is available'''
        (w,h) = cuav_util.image_shape(self.small_thumbnail)
        cv.Rectangle(self.small_thumbnail, (w-3,0), (w-1,2), color, 2) 

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
        self.shape = None

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

        if (x2-x1) > thumb_size or (y2-y1) > thumb_size:
            # we need to shrink the region
            rsize = max(x2+1-x1, y2+1-y1)
            src = cuav_util.SubImage(img, (midx-rsize/2,midy-rsize/2,rsize,rsize))
            thumb = cv.CreateImage((thumb_size, thumb_size),8,3)
            cv.Resize(src, thumb)
        else:
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
                 grid_width=20, grid_height=20, thumb_size=35, C=CameraParams(),
                 camera_settings = None,
                 image_settings = None,
                 start_menu=False,
                 classify=None):
        self.thumb_size = thumb_size
        self.width = grid_width * thumb_size
        self.height = grid_height * thumb_size
        self.mosaic = cv.CreateImage((self.height,self.width),8,3)
        cuav_util.zero_image(self.mosaic)
        self.display_regions = grid_width*grid_height
        self.regions = []
        self.regions_sorted = []
        self.regions_hidden = set()
        self.mouse_region = None
        self.ridx_by_frame_time = {}
        self.page = 0
        self.sort_type = 'Time'
        self.images = []
        self.current_view = 0
        self.last_view_latlon = None
        self.view_filename = None
        self.full_res = False
        self.boundary = []
        self.displayed_image = None
        self.last_click_position = None
        self.c_params = C
        self.camera_settings = camera_settings
        self.image_settings = image_settings
        self.start_menu = start_menu
        self.classify = classify
        self.has_started = not start_menu
        import wx
        self.image_mosaic = mp_image.MPImage(title='Mosaic', 
                                             mouse_events=True,
                                             key_events=True,
                                             auto_size=False,
                                             report_size_changes=True)
        self.slipmap = slipmap
        self.selected_region = 0

        self.view_image = None
        self.brightness = 1

        # dictionary of image requests, contains True if fullres image is wanted
        self.image_requests = {}

        self.slipmap.add_callback(functools.partial(self.map_callback))

        if classify:
            import lxml.objectify, lxml.etree
            with open(classify) as f:
                categories = lxml.objectify.fromstring(f.read())
                cat_names = set()
                self.categories = []
                try:
                    for c in categories.category:
                        self.categories.append((c.get('shortcut') or '', c.text))
                        if c.text in cat_names:
                            print 'WARNING: category name',c.text,'used more than once'
                        else:
                            cat_names.add(c.text)
                except AttributeError as ex:
                    print ex
                    print 'failed to load any categories for classification'
            self.region_class = lxml.objectify.E.regions()

        self.add_menus()
        
    def add_menus(self):
        '''add menus'''
        menu = MPMenuTop([])
        if self.start_menu:
            menu.add(MPMenuSubMenu('GEOSearch',
                                   items=[MPMenuItem('Start', 'Start', 'menuStart'),
                                          MPMenuItem('Stop', 'Stop', 'menuStop')]))
        view_menu = MPMenuSubMenu('View',
                                  items=[MPMenuRadio('Sort By', 'Select sorting key',
                                                     returnkey='setSort',
                                                     selected=self.sort_type,
                                                     items=['Score\tAlt+S',
                                                            'ScoreReverse\tAlt+R',
                                                            'Compactness\tAlt+C',
                                                            'Distinctiveness\tAlt+D',
                                                            'Whiteness\tAlt+W',
                                                            'Time\tAlt+T']),
                                         MPMenuItem('Next Page\tCtrl+N', 'Next Page', 'nextPage'),
                                         MPMenuItem('Previous Page\tCtrl+P', 'Previous Page', 'previousPage'),
                                         MPMenuItem('Brightness +\tCtrl+B', 'Increase Brightness', 'increaseBrightness'),
                                         MPMenuItem('Brightness -\tCtrl+Shift+B', 'Decrease Brightness', 'decreaseBrightness')
                                         ])
        menu.add(view_menu)
        if self.camera_settings:
            menu.add(MPMenuSubMenu('Camera',
                                   items=[MPMenuItem('Settings', 'Settings', 'menuCameraSettings')]))
        if self.image_settings:
            menu.add(MPMenuSubMenu('Image',
                                   items=[MPMenuItem('Settings', 'Settings', 'menuImageSettings')]))

        if self.classify:
            menu.add(MPMenuSubMenu('Classify',
                                   items=[MPMenuItem('{cat}\t{key}'.format(cat=cat,key=key), cat, 'classify') for (key,cat) in self.categories]))

        self.menu = menu
        self.image_mosaic.set_menu(self.menu)

        self.popup_menu = MPMenuSubMenu('Popup',
                                        items=[MPMenuItem('Show Image', returnkey='showImage'),
                                               MPMenuItem('Fetch Image', returnkey='fetchImage'),
                                               MPMenuItem('Fetch Image (full)', returnkey='fetchImageFull'),
                                               MPMenuItem('Hide Page', returnkey='hidePage'),
                                               MPMenuItem('Unhide All Pages', returnkey='unhideAll'),
                                               view_menu])
        self.image_mosaic.set_popup_menu(self.popup_menu)

    def set_mosaic_size(self, size):
        '''change mosaic size'''
        (self.width, self.height) = size
        # take into account the menu
        if self.height > 25:
            self.height -= 25
        grid_width = self.width // self.thumb_size
        if grid_width < 1:
            grid_width = 1
        grid_height = self.height // self.thumb_size
        if grid_height < 1:
            grid_height = 1
        ridx = self.page * self.display_regions
        self.display_regions = grid_width * grid_height
        self.page = ridx / self.display_regions
        self.redisplay_mosaic()

    def set_brightness(self, b):
        '''set mosaic brightness'''
        if b != self.brightness:
            self.brightness = b
            self.redisplay_mosaic()

    def show_region(self, ridx, view_the_image=False):
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
        region_text = "Selected region %u score=%u/%u/%.2f %s\n%s alt=%u yaw=%d\n%s" % (ridx, region.score,
                                                                                        region.region.scan_score,
                                                                                        region.region.compactness,
                                                                                        region.region.center(),
                                                                                        str(region.latlon),
                                                                                        region.pos.altitude, region.pos.yaw,
                                                                                        os.path.basename(region.filename))
        self.slipmap.add_object(mp_slipmap.SlipInfoText('region detail text', region_text))
        if view_the_image and os.path.exists(region.filename):
            self.view_imagefile(region.filename)

    def view_imagefile(self, filename):
        '''view an image in a zoomable window'''
        img = cuav_util.LoadImage(filename, rotate180=self.camera_settings.rotate180)
        (w,h) = cuav_util.image_shape(img)
        for i in range(len(self.images)):
            if filename == self.images[i].filename:
                self.current_view = i
                self.last_view_latlon = None
                self.images[i].shape = (w,h)
        for r in self.regions:
            if r.filename == filename:
                r.region.draw_rectangle(img, colour=(255,0,0), linewidth=min(max(w/600,1),3), offset=max(w/200,1))
        if self.view_image is None or not self.view_image.is_alive():
            import wx
            self.view_image = mp_image.MPImage(title='View',
                                               mouse_events=True,
                                               key_events=True,
                                               can_zoom=True,
                                               can_drag=True)
            vmenu = MPMenuSubMenu('View',
                                  items=[
                MPMenuItem('Next Image\tCtrl+N', 'Next Image', 'nextImage'),
                MPMenuItem('Previous Image\tCtrl+P', 'Previous Image', 'previousImage'),
                MPMenuItem('Fit Window\tCtrl+F', 'Fit Window', 'fitWindow'),
                MPMenuItem('Full Zoom\tCtrl+Z', 'Full Zoom', 'fullSize'),
                MPMenuItem('Brightness +\tCtrl+B', 'Increase Brightness', 'increaseBrightness'),
                MPMenuItem('Brightness -\tCtrl+Shift+B', 'Decrease Brightness', 'decreaseBrightness'),
                MPMenuItem('Refresh Image\tCtrl+R', 'Refresh Image', 'refreshImage'),
                MPMenuItem('Download Full\tCtrl+D', 'Download Full', 'downloadFull'),
                MPMenuItem('Place Marker\tCtrl+M', 'Place Marker', 'placeMarker')])
            self.view_menu = MPMenuTop([vmenu])
            self.view_image.set_menu(self.view_menu)
            self.view_image.set_popup_menu(vmenu)
        self.view_filename = filename
        self.view_image.set_image(img, bgr=True)
        self.view_image.set_title('View: ' + os.path.basename(filename))

    def find_image_idx(self, filename):
        '''find index of image'''
        for i in range(len(self.images)):
            if self.images[i].filename == filename:
                return i
        return None

    def view_imagefile_by_idx(self, idx):
        '''view an image in a zoomable window by index'''
        if idx is None or idx < 0 or idx >= len(self.images):
            return
        self.view_imagefile(self.images[idx].filename)

    def show_selected(self, selected):
        '''try to show a selected image'''
        key = str(selected.objkey)
        if not key.startswith("region "):
            return False
        r = key.split()
        ridx = int(r[1])
        print("Selected %s ridx=%u" % (key, ridx))
        if ridx < 0 or ridx >= len(self.regions):
            print("Invalid region %u selected" % ridx)
            return False
        region = self.regions[ridx]
        if os.path.exists(region.filename):
            self.view_imagefile(region.filename)
            return True
            
        return False

    def show_closest(self, latlon, selected):
        '''show closest camera image'''
        # first try to show the exact image selected
        if len(selected) != 0 and self.show_selected(selected[0]):
            return
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
        self.last_view_latlon = None
        image = self.images[closest]
        self.view_imagefile(image.filename)

    def map_menu_callback(self, event):
        '''called on popup menu on map'''
        menuitem = event.menuitem
        if menuitem.returnkey == 'showImage':
            region = self.objkey_to_region(event.selected[0].objkey)
            self.popup_show_image(region)
        elif menuitem.returnkey in ['fetchImage', 'fetchImageFull']:
            region = self.objkey_to_region(event.selected[0].objkey)
            self.popup_fetch_image(region, menuitem.returnkey)

    def map_callback(self, event):
        '''called when an event happens on the slipmap'''
        if isinstance(event, mp_slipmap.SlipMenuEvent):
            self.map_menu_callback(event)
            return
        if not isinstance(event, mp_slipmap.SlipMouseEvent):
            return
        if event.event.m_middleDown:
            # show closest image from history
            self.show_closest(event.latlon, event.selected)
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


    def hide_page(self):
        '''hide a page of thumbnails in mosaic'''
        first = self.display_regions * self.page
        count = self.display_regions
        if first + count >= len(self.regions_sorted):
            count = len(self.regions_sorted) - first
        print("hiding %u regions starting at %u" % (count, first))
        for i in range(count):
            r = self.regions_sorted.pop(first)
            self.regions_hidden.add(r.ridx)
            self.slipmap.hide_object("region %u" % r.ridx)
        self.redisplay_mosaic()
        self.change_page(self.page)

    def unhide_all(self):
        '''unhide all pages in mosaic'''
        for ridx in self.regions_hidden:
            self.regions_sorted.append(self.regions[ridx])
            self.slipmap.hide_object("region %u" % ridx, hide=False)
        self.regions_hidden = set()
        self.redisplay_mosaic()

    def change_page(self, page):
        '''change page number'''
        last_page = self.page
        self.page = page
        if self.page < 0:
            self.page = 0
        max_page = (len(self.regions_sorted)-1) / self.display_regions
        if max_page < 0:
            max_page = 0
        if self.page > max_page:
            self.page = max_page
        if last_page != self.page:
            print("Page %u/%u" % (self.page, max_page))
            self.redisplay_mosaic()
        self.image_mosaic.set_title("Mosaic (Page %u of %u)" % (self.page+1, max(max_page+1, 1)))

    def re_sort(self):
        '''re sort the mosaic'''
        print("Sorting by %s" % self.sort_type)
        sortby = self.sort_type
        if sortby == 'Score':
            self.regions_sorted.sort(key = lambda r : r.score, reverse=True)
        elif sortby == 'ScoreReverse':
            self.regions_sorted.sort(key = lambda r : r.score, reverse=False)
        elif sortby == 'Compactness':
            self.regions_sorted.sort(key = lambda r : r.region.compactness, reverse=True)
        elif sortby == 'Distinctiveness':
            self.regions_sorted.sort(key = lambda r : r.region.scan_score, reverse=True)
        elif sortby == 'Whiteness':
            self.regions_sorted.sort(key = lambda r : r.region.whiteness, reverse=True)
        elif sortby == 'Time':
            self.regions_sorted.sort(key = lambda r : r.ridx, reverse=False)
        else:
            print("Unknown sort by '%s'" % sortby)
            
    def menu_event(self, event):
        '''called on menu events on the mosaic'''
        if event.returnkey == 'setSort':
            sortby = event.get_choice()
            sortby = sortby.split('\t')[0]
            self.sort_type = sortby
            self.re_sort()
            self.redisplay_mosaic()
        elif event.returnkey == 'nextPage':
            self.change_page(self.page + 1)
        elif event.returnkey == 'previousPage':
            self.change_page(self.page - 1)
        elif event.returnkey == 'increaseBrightness':
            self.set_brightness(self.brightness * 1.25)
        elif event.returnkey == 'decreaseBrightness':
            self.set_brightness(self.brightness / 1.25)
        elif event.returnkey == 'showImage':
            region = self.pos_to_region(event.popup_pos)
            self.popup_show_image(region)
        elif event.returnkey in ['fetchImage', 'fetchImageFull']:
            region = self.pos_to_region(event.popup_pos)
            self.popup_fetch_image(region, event.returnkey)
        elif event.returnkey == 'menuCameraSettings':
            WXSettings(self.camera_settings)
        elif event.returnkey == 'menuImageSettings':
            WXSettings(self.image_settings)
        elif event.returnkey == 'menuStart':
            self.has_started = True
        elif event.returnkey == 'menuStop':
            self.has_started = False
        elif event.returnkey == 'hidePage':
            self.hide_page()
        elif event.returnkey == 'unhideAll':
            self.unhide_all()
        elif event.returnkey == 'classify':
            r = self.mouse_region
            if r is None:
                return
            filename = r.filename
            r = r.region
            import lxml.objectify, lxml.etree
            E = lxml.objectify.E
            self.region_class.append(
                E.region(E.filename(filename), E.x1(r.x1), E.y1(r.y1), E.x2(r.x2), E.y2(r.y2), E.category(event.description))
            )
            with open('regions.xml', 'w') as f:
                # this will become kind of inefficient;
                # consider only writing file once at the end?
                f.write(lxml.etree.tostring(self.region_class, pretty_print=True))

            try:
                # highlight active region
                old_ridx = self.mouse_region.ridx
                self.mouse_region = self.regions_sorted[old_ridx+1]
                self.display_mosaic_region(old_ridx)
                self.display_mosaic_region(self.mouse_region.ridx)
                self.redisplay_mosaic()
                self.show_region(self.mouse_region.ridx, False)
            except IndexError as e:
                # no more thumbnails to classify
                self.mouse_region = None

    def started(self):
        '''return if start button has been pushed'''
        return self.has_started

    def popup_show_image(self, region):
        '''handle popup menu showImage'''
        if region is None:
            return
        self.show_region(region.ridx, True)
        if region.latlon != (None,None):
            self.slipmap.add_object(mp_slipmap.SlipCenter(region.latlon))

    def popup_fetch_image(self, region, returnkey):
        '''handle popup menu fetchImage'''
        if region is None:
            return
        fullres = (returnkey == 'fetchImageFull')
        frame_time = cuav_util.parse_frame_time(region.filename)
        self.image_requests[frame_time] = fullres

    def get_image_requests(self):
        '''return and zero image_requests dictionary'''
        ret = self.image_requests
        self.image_requests = {}
        return ret

    def menu_event_view(self, event):
        '''called on menu events on the view image'''
        if event.returnkey == 'increaseBrightness':
            self.set_brightness(self.brightness * 1.25)
            self.view_image.set_brightness(self.brightness)
        elif event.returnkey == 'decreaseBrightness':
            self.set_brightness(self.brightness / 1.25)
            self.view_image.set_brightness(self.brightness)
        elif event.returnkey == 'fitWindow':
            self.view_image.fit_to_window()
        elif event.returnkey == 'fullSize':
            self.view_image.full_size()
        elif event.returnkey == 'nextImage':
            idx = self.find_image_idx(self.view_filename)
            if idx is not None:
                self.view_imagefile_by_idx(idx+1)
        elif event.returnkey == 'refreshImage':
            idx = self.find_image_idx(self.view_filename)
            if idx is not None:
                self.view_imagefile_by_idx(idx)
        elif event.returnkey == 'downloadFull':
            idx = self.find_image_idx(self.view_filename)
            if idx is not None:
                frame_time = self.images[idx].frame_time
                self.image_requests[frame_time] = True
        elif event.returnkey == 'previousImage':
            idx = self.find_image_idx(self.view_filename)
            if idx is not None:
                self.view_imagefile_by_idx(idx-1)
        elif event.returnkey == 'placeMarker':
            if self.current_view >= len(self.images):
                return
            if self.last_view_latlon is None:
                print("Please left click first")
                return
            image = self.images[self.current_view]
            latlon = self.last_view_latlon
            if latlon is None:
                return
            icon = self.slipmap.icon('flag.png')
            self.slipmap.add_object(mp_slipmap.SlipIcon('Marker-%u' % self.current_view,
                                                        latlon=latlon,
                                                        layer='Markers',
                                                        img=icon,
                                                        follow=False))
                

    def pos_to_region(self, pos):
        '''work out region for a clicked position on the mosaic'''
        x = pos.x
        y = pos.y
        page_idx = (x/self.thumb_size) + (self.width/self.thumb_size)*(y/self.thumb_size)
        ridx = page_idx + self.page * self.display_regions
        if ridx < 0 or ridx >= len(self.regions_sorted):
            return None
        return self.regions_sorted[ridx]

    def objkey_to_region(self, objkey):
        '''work out region for a map objkey'''
        if not objkey.startswith("region "):
            return None
        ridx = int(objkey[7:])
        if ridx < 0 or ridx >= len(self.regions):
            return None
        return self.regions[ridx]

    def mouse_event(self, event):
        '''called on mouse events on the mosaic'''
        import wx
        if event.X < 0 or event.Y < 0:
            # sometimes get events when the mouse cursor is not on the mosaic
            return
        #print 'cuav_mosaic mouse_event',event.__dict__

        # work out which region they want, taking into account wrap
        region = self.pos_to_region(wx.Point(event.X, event.Y))
        if region is None:
            return

        if event.m_leftDown: # TODO is this dangerous
            self.show_region(region.ridx, event.m_middleDown)
            if region.latlon != (None,None):
                self.slipmap.add_object(mp_slipmap.SlipCenter(region.latlon))
        else:
            # highlight on mouseover
            old_region = self.mouse_region
            self.mouse_region = region
            self.display_mosaic_region(region.ridx)
            if old_region != None:
                self.display_mosaic_region(old_region.ridx)
            if not self.started():
                # if the search is started, it'll be redisplayed anyway
                self.redisplay_mosaic()


    def mouse_event_view(self, event):
        '''called on mouse events in View window'''
        x = event.X
        y = event.Y
        if self.current_view >= len(self.images):
            return
        image = self.images[self.current_view]
        latlon = cuav_util.gps_position_from_xy(x, y, image.pos, C=self.c_params, shape=image.shape)
        if self.last_view_latlon is None or latlon is None:
            dist = ''
        else:
            distance = cuav_util.gps_distance(self.last_view_latlon[0], self.last_view_latlon[1],
                                              latlon[0], latlon[1])
            bearing = cuav_util.gps_bearing(self.last_view_latlon[0], self.last_view_latlon[1],
                                             latlon[0], latlon[1])
            dist = "dist %.1f bearing %.1f alt=%.1f shape=%s" % (distance, bearing, image.pos.altitude, image.shape)
        print("-> %s %s %s" % (latlon, image.filename, dist))
        self.last_view_latlon = latlon

    def key_event(self, event):
        '''called on key events'''
        pass

    def region_on_page(self, ridx, page):
        '''return True if a region is on the given page.
        ridx is an index into regions_sorted[]'''
        if ridx < 0 or ridx >= len(self.regions_sorted):
            return False
        width = (self.width // self.thumb_size) * self.thumb_size
        page_idx = ridx - page * self.display_regions
        if page_idx < 0 or page_idx >= self.display_regions:
            # its not on this page
            return False
        return True
    
    def display_mosaic_region(self, ridx):
        '''display a thumbnail on the mosaic'''
        if not self.region_on_page(ridx, self.page):
            return
        region = self.regions_sorted[ridx]
        width = (self.width // self.thumb_size) * self.thumb_size
        page_idx = ridx - self.page * self.display_regions
        dest_x = (page_idx * self.thumb_size) % width
        dest_y = ((page_idx * self.thumb_size) / width) * self.thumb_size

        if region == self.mouse_region:
            thumb = cv.CreateImage((self.thumb_size,self.thumb_size), 8, 3)
            cv.ConvertScale(region.small_thumbnail, thumb, 2.0)
        else:
            thumb = region.small_thumbnail

        # overlay thumbnail on mosaic
        #print dest_x, dest_y, self.width, self.height, self.thumb_size, cuav_util.image_width(region.small_thumbnail)
        try:
            cuav_util.OverlayImage(self.mosaic, thumb, dest_x, dest_y)
        except Exception:
            pass

    def redisplay_mosaic(self):
        '''re-display whole mosaic page'''
        width = (self.width // self.thumb_size) * self.thumb_size
        height = (self.height // self.thumb_size) * self.thumb_size
        self.mosaic = cv.CreateImage((width,height),8,3)
        cuav_util.zero_image(self.mosaic)
        for ridx in range(len(self.regions_sorted)):
            self.display_mosaic_region(ridx)
        if self.brightness != 1.0:
            cv.ConvertScale(self.mosaic, self.mosaic, scale=self.brightness)
        self.image_mosaic.set_image(self.mosaic, bgr=True)
        
        max_page = (len(self.regions_sorted)-1) / self.display_regions
        self.image_mosaic.set_title("Mosaic (Page %u of %u)" % (self.page+1, max(max_page+1, 1)))

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
            rsize = max(x2+1-x1,y2+1-y1)
            tsize = cuav_util.image_width(full_thumb)
            if rsize < tsize:
                thumb = cuav_util.SubImage(full_thumb, ((tsize-self.thumb_size)//2,
                                                        (tsize-self.thumb_size)//2,
                                                        self.thumb_size,
                                                        self.thumb_size))
            else:
                thumb = cv.CreateImage((self.thumb_size, self.thumb_size),8,3)
                cv.Resize(full_thumb, thumb)

            ridx = len(self.regions)
            self.regions.append(MosaicRegion(ridx, r, filename, pos, thumbs[i], thumb, latlon=(lat,lon)))
            self.regions_sorted.append(self.regions[-1])
            
            max_page = (len(self.regions_sorted)-1) / self.display_regions
            self.image_mosaic.set_title("Mosaic (Page %u of %u)" % (self.page+1, max(max_page+1, 1)))

            frame_time = cuav_util.parse_frame_time(filename)
            if not frame_time in self.ridx_by_frame_time:
                self.ridx_by_frame_time[frame_time] = [ridx]
            else:
                self.ridx_by_frame_time[frame_time].append(ridx)

            self.display_mosaic_region(len(self.regions_sorted)-1)

            if (lat,lon) != (None,None):
                self.slipmap.add_object(mp_slipmap.SlipThumbnail("region %u" % ridx, (lat,lon),
                                                                 img=thumb,
                                                                 layer=2, border_width=1, border_colour=(255,0,0),
                                                                 popup_menu=self.popup_menu))

        self.image_mosaic.set_image(self.mosaic, bgr=True)

    def add_image(self, frame_time, filename, pos):
        '''add a camera image'''
        idx = self.find_image_idx(filename)
        if idx is not None:
            self.images[idx].pos = pos
            self.images[idx].frame_time = frame_time
        else:
            self.images.append(MosaicImage(frame_time, filename, pos))

    def tag_image(self, frame_time, tag_color=(0,255,255)):
        '''tag a mosaic image'''
        tagged = False
        if frame_time in self.ridx_by_frame_time:
            for ridx in self.ridx_by_frame_time[frame_time]:
                self.regions[ridx].tag_image_available(color=tag_color)
                tagged = True
        if tagged:
            self.redisplay_mosaic()

    def check_events(self):
        '''check for mouse/keyboard events'''
        if self.image_mosaic.is_alive():
            for event in self.image_mosaic.events():
                if isinstance(event, mp_image.MPImageNewSize):
                    self.set_mosaic_size(event.size)
                elif isinstance(event, MPMenuGeneric):
                    self.menu_event(event)
                elif event.ClassName == 'wxMouseEvent':
                    self.mouse_event(event)
                elif event.ClassName == 'wxKeyEvent':
                    self.key_event(event)
                else:
                    print('unknown event ', event)
                    print(dir(event))
        if self.view_image and self.view_image.is_alive():
            for event in self.view_image.events():
                if isinstance(event, MPMenuGeneric):
                    self.menu_event_view(event)
                elif event.ClassName == 'wxMouseEvent':
                    self.mouse_event_view(event)
            
        
