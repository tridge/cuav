#!/usr/bin/env python
'''Class for determining and plotting a landing location for a UAV, based on a set
of detected regions.'''

import cuav_util

class LandingZoneDisplay:
    '''this is a landing zone object for transmitting to the GCS for display purposes'''
    def __init__(self, latlon, maxrange, avgscore, numregions):
        self.latlon = latlon
        self.maxrange = maxrange
        self.avgscore = avgscore
        self.numregions = numregions

class LandingZone:
    def __init__(self):
        self.regions = []

    def checkaddregion(self, r, pos):
        '''Add a region to the list of landing zone regions'''
        # remember the bank angle and yaw for weighting
        r.angle = pos.roll + pos.pitch
        r.yaw = pos.yaw
        self.regions.append(r)

    def distance_from(self, region, center):
        '''return distance of a region from a center point'''
        (lat, lon) = region.latlon
        (clat, clon) = center
        return cuav_util.gps_distance(lat, lon, clat, clon)

    def average_pos(self, regions):
        '''return (lat,lon) average of a list of regions'''
        lat_sum = 0.0
        lon_sum = 0.0
        for r in regions:
            (lat,lon) = r.latlon
            lat_sum += lat
            lon_sum += lon
        return (lat_sum / len(regions), lon_sum / len(regions))

    def calclandingzone(self):
        '''work out best estimate of the landing zone
        
        We want to average the regions with the following constraints:

         *) we want regions with a high target score
         *) we want regions with low roll/pitch of the aircraft for higher accuracy
         *) we want regions which have a wide range of yaw values
         *) we want to eliminate outliers
        '''
        bestlz = None

        # we must have at least 10 samples
        if len(self.regions) < 10:
            return None
        
        # start by dropping the bottom 25% percentile by score. This removes
        # the likely bad matches

        # take a shallow copy
        regions = self.regions[:]

        # throw away bottom 25% by score
        regions.sort(key = lambda r : r.score, reverse=True)
        regions = regions[:-len(regions)/4]

        # throw away bottom 25% by angle
        regions.sort(key = lambda r : r.angle, reverse=False)
        regions = regions[:-len(regions)/4]

        # find average position
        center = self.average_pos(regions)

        # throw away furthest 20%, to discard outliers
        regions.sort(key = lambda r : self.distance_from(r, center), reverse=False)
        regions = regions[:-len(regions)/5]

        # re-calculate center
        center = self.average_pos(regions)

        maxerror = 0.0
        sumscore = 0.0
        for r in regions:
            d = self.distance_from(r, center)
            maxerror = max(d, maxerror)
            sumscore += r.score

        return LandingZoneDisplay(center, maxerror, sumscore / len(regions), len(regions))
