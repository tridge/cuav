#!/usr/bin/python
'''
display a set of found regions as a mosaic
Andrew Tridgell
May 2012
'''

import numpy, os, cv, sys, cuav_util, time, math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'image'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy', 'modules', 'lib'))
import scanner, mp_image, mp_slipmap

def compute_signatures(hist1, hist2):
    '''
    demos how to convert 2 histograms into 2 signature
    '''
    h_bins = hist1.bins.shape(0)
    s_bins = hist1.bins.shape(1)
    num_rows = h_bins * s_bins
    sig1 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    sig2 = cv.CreateMat(num_rows, 3, cv.CV_32FC1)
    #fill signatures
    #TODO: for production optimize this, use Numpy
    for h in range(0, h_bins):
        for s in range(0, s_bins):
            bin_val = cv.QueryHistValue_2D(hist1, h, s)
            cv.Set2D(sig1, h*s_bins + s, 0, bin_val) #bin value
            cv.Set2D(sig1, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig1, h*s_bins + s, 2, s) #coord2
            #signature.2
            bin_val2 = cv.QueryHistValue_2D(hist2, h, s)
            cv.Set2D(sig2, h*s_bins + s, 0, bin_val2) #bin value
            cv.Set2D(sig2, h*s_bins + s, 1, h)  #coord1
            cv.Set2D(sig2, h*s_bins + s, 2, s) #coord2

    return (sig1, sig2)

class MosaicRegion:
    def __init__(self, region, filename, pos, thumbnail, latlon=(None,None), map_pos=(None,None)):
        # self.region is a (minx,miny,maxy,maxy) rectange in image coordinates
        self.region = region
        self.filename = filename
        self.pos = pos
        self.thumbnail = thumbnail
        self.map_pos = map_pos
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

def CompositeThumbnail(img, regions, thumb_size=100, quality=75):
    '''extract a composite thumbnail for the regions of an image

    The composite will consist of N thumbnails side by side

    return it as a compressed jpeg string
    '''
    composite = numpy.zeros((thumb_size, thumb_size*len(regions),3),dtype='uint8')
    thumb = numpy.zeros((thumb_size, thumb_size,3),dtype='uint8')
    for i in range(len(regions)):
        (x1,y1,x2,y2) = regions[i]
        midx = (x1+x2)/2
        midy = (y1+y2)/2

        if cuav_util.image_width(img) == 1280:
            # the regions are from a 640x480 image. If we are extracting
            # from a 1280x960, then move the central pixel
            midx *= 2
            midy *= 2

        x1 = max(midx - thumb_size/2, 0)
        y1 = max(midy - thumb_size/2, 0)
        scanner.rect_extract(img, thumb, x1, y1)
        scanner.rect_overlay(composite, thumb, thumb_size*i, 0, True)
    return scanner.jpeg_compress(composite, quality)

def ExtractThumbs(img, count):
    '''extract thumbnails from a composite thumbnail image'''
    thumb_size = cuav_util.image_width(img) / count
    img = numpy.asarray(cv.GetMat(img))
    thumbs = []
    for i in range(count):
        thumb = numpy.zeros((thumb_size, thumb_size,3),dtype='uint8')
        scanner.rect_extract(img, thumb, i*thumb_size, 0)
        thumbs.append(thumb)
    return thumbs

class Mosaic():
    '''keep a mosaic of found regions'''
    def __init__(self, slipmap,
                 grid_width=20, grid_height=20, thumb_size=30, lens=4.0, fill_map=True):
        self.thumb_size = thumb_size
        self.width = grid_width * thumb_size
        self.height = grid_height * thumb_size
        self.map_width = grid_width * thumb_size
        self.map_height = grid_height * thumb_size
        self.mosaic = numpy.zeros((self.height,self.width,3),dtype='uint8')
        self.map = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
        self.map_background = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
        self.display_regions = grid_width*grid_height
        self.regions = []
        self.images = []
        self.full_res = False
        self.boundary = []
        self.fill_map = fill_map
        self.last_map_image_idx = None
        self.displayed_image = None
        self.last_click_position = None
        self.lens = lens
        self.image_mosaic = mp_image.MPImage()
        self.slipmap = slipmap

        # map limits in form (min_lat, min_lon, max_lat, max_lon)
        self.map_limits = [0,0,0,0]



    def latlon_to_map(self, lat, lon):
        '''
        convert a latitude/longitude to map position in pixels, with
        0,0 in top left
        '''
        dlat = lat - self.map_limits[0]
        dlon = lon - self.map_limits[1]
        px = (dlon / (self.map_limits[3] - self.map_limits[1])) * self.map_width
        py = (1.0 - dlat / (self.map_limits[2] - self.map_limits[0])) * self.map_height
        return (int(px+0.5), int(py+0.5))

    def map_to_latlon(self, x, y):
        '''
        convert a x/y map position in pixels to a lat/lon
        '''
        lat = self.map_limits[0] + ((self.map_height-1)-y)*(self.map_limits[2] - self.map_limits[0])/(self.map_height-1)
        lon = self.map_limits[1] + x*(self.map_limits[3] - self.map_limits[1])/(self.map_width-1)
        return (lat, lon)


    def set_boundary(self, boundary):
        '''set a polygon search boundary'''
        if not cuav_util.polygon_complete(boundary):
            raise RuntimeError('invalid boundary passed to mosaic')
        self.boundary = boundary[:]
        self.map_limits = [boundary[0][0], boundary[0][1],
                           boundary[0][0], boundary[0][1]]
        for b in self.boundary:
            (lat,lon) = b
            self.map_limits[0] = min(self.map_limits[0], lat)
            self.map_limits[1] = min(self.map_limits[1], lon)
            self.map_limits[2] = max(self.map_limits[2], lat)
            self.map_limits[3] = max(self.map_limits[3], lon)

        # add a 50m border
        (lat, lon) = cuav_util.gps_newpos(self.map_limits[0], self.map_limits[1],
                                          225, 50)
        self.map_limits[0] = lat
        self.map_limits[1] = lon

        (lat, lon) = cuav_util.gps_newpos(self.map_limits[2], self.map_limits[3],
                                          45, 50)
        self.map_limits[2] = lat
        self.map_limits[3] = lon

        # draw the border
        img = cv.GetImage(cv.fromarray(self.map))
        for i in range(len(self.boundary)-1):
            (lat1,lon1) = self.boundary[i]
            (lat2,lon2) = self.boundary[i+1]
            cv.Line(img,
                    self.latlon_to_map(lat1, lon1),
                    self.latlon_to_map(lat2, lon2),
                    (0,255,0), 3)
        self.map = numpy.asarray(cv.GetMat(img))
        self.slipmap.add_object(mp_slipmap.SlipPolygon('boundary', self.boundary, layer=1, linewidth=2, colour=(0,255,0)))


    def histogram(self, image, bins=10):
        '''return a 2D HS histogram of an image'''
        mat = cv.fromarray(image)
        hsv = cv.CreateImage(cv.GetSize(mat), 8, 3)
        cv.CvtColor(mat, hsv, cv.CV_BGR2Lab)
        planes = [ cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC1),
                   cv.CreateMat(mat.rows, mat.cols, cv.CV_8UC1) ]
        cv.Split(hsv, planes[0], planes[1], None, None)
        hist = cv.CreateHist([bins, bins], cv.CV_HIST_ARRAY)
        cv.CalcHist([cv.GetImage(p) for p in planes], hist)
        cv.NormalizeHist(hist, 1024)
        return hist

    def show_hist(self, hist):
        print(len(hist.bins))
        bins = len(hist.bins)
        a = numpy.zeros((bins, bins), dtype='uint16')
        for h in range(bins):
            for s in range(bins):
                a[h,s] = cv.QueryHistValue_2D(hist, h, s)
        print a


    def measure_distinctiveness(self, image, r):
        '''measure the distinctiveness of a region in an image
        this is currently disabled, and needs more work
        '''
        return
        if cuav_util.image_width(image) == 1280:
            r = tuple([2*v for v in r])
        (x1,y1,x2,y2) = r

        hist1 = self.histogram(image)
        self.show_hist(hist1)

        thumbnail = numpy.zeros((y2-y1+1, x2-x1+1,3),dtype='uint8')
        scanner.rect_extract(image, thumbnail, x1, y1)

        hist2 = self.histogram(thumbnail)
        self.show_hist(hist2)

        (sig1, sig2) = compute_signatures(hist1, hist2, h_bins=32, s_bins=32)
        return cv.CalcEMD2(sig1, sig2, cv.CV_DIST_L2, lower_bound=float('inf'))

        print cv.CompareHist(hist1, hist2, cv.CV_COMP_CHISQR)

    def display_region_image(self, region):
        '''display the thumbnail associated with a region'''
        print('-> %s' % str(region))
        img = cv.fromarray(region.thumbnail)
#      cv.ShowImage('Thumb', img)
#      cv.WaitKey(1)


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


    def find_closest_region(self, x, y):
        '''find the closest region given a pixel position on the map'''
        if len(self.regions) == 0:
            return None
        best_idx = 0
        best_distance = -1
        for idx in range(len(self.regions)):
            if self.regions[idx].map_pos == (None,None):
                continue
            distance = math.sqrt((self.regions[idx].map_pos[0] - x)**2 + (self.regions[idx].map_pos[1] - y)**2)
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_distance == -1 or best_distance > self.thumb_size:
            return None
        return self.regions[best_idx]

    def find_closest_image_idx(self, x, y):
        '''find the closest image given a pixel position on the map'''
        if len(self.images) == 0:
            return None
        best_idx = 0
        best_distance = -1
        (lat, lon) = self.map_to_latlon(x,y)
        for idx in range(len(self.images)):
            if self.images[idx].center == (None,None):
                continue
            if cuav_util.polygon_outside((lat, lon), self.images[idx].boundary):
                # only include images where we clicked inside the boundary
                continue
            distance = cuav_util.gps_distance(lat, lon, self.images[idx].center[0], self.images[idx].center[1])
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_distance == -1:
            return None
        return best_idx

    def mouse_event_map(self, event, x, y, flags, data):
        '''called on mouse events on the map'''
        if flags & cv.CV_EVENT_FLAG_RBUTTON:
            self.full_res = not self.full_res
            print("full_res: %s" % self.full_res)

        if flags & cv.CV_EVENT_FLAG_LBUTTON:
            # find the closest marked region and display it
            region = self.find_closest_region(x,y)
            if region is None:
                return    
            self.display_region_image(region)

        if flags & cv.CV_EVENT_FLAG_MBUTTON:
            # find the closest image re-display it
            idx = self.find_closest_image_idx(x,y)
            if idx is None:
                return    
            if self.last_map_image_idx != idx:
                self.last_map_image_idx = idx
                self.display_map_image(self.images[idx], show_full=True)


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
                                             lens=self.lens)
        if latlon is None:
            print("Unable to find pixel coordinates")
            return
        (lat,lon) = latlon
        print("=> %s %f %f %s" % (img.filename, lat, lon, str(pos)))
        if self.last_click_position:
            (last_lat, last_lon) = self.last_click_position
            print("distance from last click: %.1f m" % (cuav_util.gps_distance(lat, lon, last_lat, last_lon)))
        self.last_click_position = (lat,lon)


    def refresh_map(self):
        '''refresh the map display'''
        if self.fill_map:
            map = numpy.zeros((self.map_height,self.map_width,3),dtype='uint8')
            scanner.rect_overlay(map, self.map_background, 0, 0, False)
            scanner.rect_overlay(map, self.map, 0, 0, True)
        else:
            map = self.map
#    cv.ShowImage('Mosaic Map', cv.fromarray(map))

    def image_boundary(self, img, pos):
        '''return a set of 4 (lat,lon) coordinates for the 4 corners
        of an image in the order
        (left,top), (right,top), (right,bottom), (left,bottom)

        Note that one or more of the corners may return as None if
        the corner is not on the ground (it points at the sky)
        '''
        (w,h) = cuav_util.image_shape(img)
        latlon = []
        for (x,y) in [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]:
            latlon.append(cuav_util.pixel_coordinates(x, y, pos.lat, pos.lon, pos.altitude,
                                                      pos.pitch, pos.roll, pos.yaw,
                                                      xresolution=w, yresolution=h,
                                                      lens=self.lens))
        # make it a complete polygon by appending the first point
        latlon.append(latlon[0])
        return latlon

    def image_center(self, img, pos):
        '''return a (lat,lon) for the center of the image
        '''
        (w,h) = cuav_util.image_shape(img)
        latlon = []
        return cuav_util.pixel_coordinates(w/2, h/2, pos.lat, pos.lon, pos.altitude,
                                           pos.pitch, pos.roll, pos.yaw,
                                           xresolution=w, yresolution=h,
                                           lens=self.lens)

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


    def display_map_image(self, image, show_full=False, showrectangle=False):
        '''show transformed image on map'''
        try:
            img = cuav_util.LoadImage(image.filename)
        except Exception:
            print("unable to LoadImage %s" % image.filename)
            return
        if show_full:
            self.displayed_image = DisplayedImage(image.filename, image.pos, img)
#      cv.ShowImage('Image', img)
#      cv.SetMouseCallback('Image', self.mouse_event_image, self)
            print('=> %s %s' % (image.filename, str(image.pos)))
        (w,h) = cuav_util.image_shape(img)
        srcpos = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
        dstpos = [self.latlon_to_map(lat, lon) for (lat,lon) in image.boundary[0:4]]
        if self.image_area(dstpos) < 10:
            return
        transform = cv.fromarray(numpy.zeros((3,3),dtype=numpy.float32))
        cv.GetPerspectiveTransform(srcpos, dstpos, transform)
        map_bg = cv.GetImage(cv.fromarray(self.map_background))
        cv.WarpPerspective(img, map_bg, transform, flags=0)
        self.map_background = numpy.asarray(cv.GetMat(map_bg))
        self.refresh_map()


    def add_image(self, filename, img, pos):
        '''add a background image'''
        if not self.fill_map:
            return
        if pos.altitude < 5:
            # don't bother with pictures of the runway
            return
        img_boundary = self.image_boundary(img, pos)
        if None in img_boundary:
            # an image corner extends into the sky
            return
        center = self.image_center(img, pos)
        self.images.append(MosaicImage(filename, pos, img_boundary, center))
        self.display_map_image(self.images[-1])

    def add_regions(self, regions, thumbs, latlon_list, filename, pos=None):
        '''add some regions'''
        for i in range(len(regions)):
            r = regions[i]
            (x1,y1,x2,y2) = r

            (mapx, mapy) = (None, None)
            (lat, lon) = latlon_list[i]

            if self.boundary and (lat,lon) == (None,None):
                # its pointing into the sky
                continue
            if self.boundary:
                if cuav_util.polygon_outside((lat, lon), self.boundary):
                    # this region is outside the search boundary
                    continue

            # the thumbnail we have been given will be bigger than the size we want to
            # display on the mosaic and map. Extract the middle of it for display
            thumb = numpy.zeros((self.thumb_size, self.thumb_size, 3), dtype='uint8')
            tsize = cuav_util.image_width(thumbs[i])
            scanner.rect_extract(thumbs[i], thumb,
                                 (tsize-self.thumb_size)//2,
                                 (tsize-self.thumb_size)//2) 

            idx = len(self.regions) % self.display_regions

            dest_x = (idx * self.thumb_size) % self.width
            dest_y = ((idx * self.thumb_size) / self.width) * self.thumb_size

            # overlay thumbnail on mosaic
            scanner.rect_overlay(self.mosaic, thumb, dest_x, dest_y, False)

            if (lat,lon) != (None,None):
                # show thumbnail on map
                (mapx, mapy) = self.latlon_to_map(lat, lon)
                scanner.rect_overlay(self.map, thumb,
                                     max(0, mapx - self.thumb_size/2),
                                     max(0, mapy - self.thumb_size/2), False)
                map = cv.fromarray(self.map)
                (x1,y1) = (max(0, mapx - self.thumb_size/2),
                           max(0, mapy - self.thumb_size/2))
                (x2,y2) = (x1+self.thumb_size, y1+self.thumb_size)
                cv.Rectangle(map, (x1,y1), (x2,y2), (255,0,0), 1)
                self.map = numpy.asarray(map)
                self.refresh_map()

                self.slipmap.add_object(mp_slipmap.SlipThumbnail((lat,lon),(lat,lon),
                                                                 img=cv.fromarray(thumb),
                                                                 layer=2, border_width=1, border_colour=(255,0,0)))

            self.regions.append(MosaicRegion(r, filename, pos, thumbs[i], latlon=(lat, lon), map_pos=(mapx, mapy)))

        self.image_mosaic.set_image(cv.GetImage(cv.fromarray(self.mosaic)), bgr=True)
#      cv.SetMouseCallback('Mosaic', self.mouse_event, self)


    def check_joe_miss(self, regions, img, joes, pos, accuracy=80):
        '''check for false negatives from Joe scanner'''

        # work out the lat,lon coordinates of the four corners of the image
        image_boundary = self.image_boundary(img, pos)

        if None in image_boundary:
            # an image corner extends into the sky
            return

        for joe in joes:
            if cuav_util.polygon_outside(joe, image_boundary):
                continue
            # this joe should be in this image
            (joe_lat, joe_lon) = joe
            min_error = -1
            for r in regions:
                latlon = cuav_util.gps_position_from_image_region(r, pos, lens=self.lens)
                if latlon is None:
                    # its in the sky
                    continue
                (lat, lon) = latlon
                if cuav_util.polygon_outside((lat,lon), image_boundary):
                    print("Found position outside image boundary??", r, str(pos), (lat,lon), image_boundary)
                error = cuav_util.gps_distance(joe_lat, joe_lon, lat, lon)
                if min_error == -1 or error < min_error:
                    min_error = error
            if min_error > accuracy:
                # we got a false negative
                print("False negative min_error=%f" % min_error)
