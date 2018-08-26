#!/usr/bin/env python
'''Class for determining and plotting a landing location for a UAV, based on a set
of detected regions.'''

import cuav_util

class RegionClump:
    '''A set of nearby regions that should be a landing zone'''
    def __init__(self, r):
        self.regions = [] #Tuple of (lon, lat, score)
        self.avgscore = 0
        (la, lo) = r.latlon
        self.regions.append((la, lo, r.score))
        self.centreloc = (la, lo)
        self.maxerror = 50

    def calccentrepoint(self):
        '''Calc the average lon/lat point
        of all the regions'''
        if len(self.regions) == 0:
            return
        else:
            avglat = 0
            avglon = 0
            for rcur in self.regions:
                (la, lo, sc) = rcur
                avglat += la
                avglon += lo
            avglat /= len(self.regions)
            avglon /= len(self.regions)
            self.centreloc = (avglat, avglon)
        #and the self.maxerror and self.avgscore
        self.maxerror = None
        self.avgscore = 0
        for rcur in self.regions:
            (la, lo, sc) = rcur
            newerror = cuav_util.gps_distance(avglat, avglon, la, lo)
            self.avgscore += sc
            if newerror > self.maxerror or self.maxerror is None:
                self.maxerror = newerror
        self.avgscore /= len(self.regions)
        #give a bad average for very small clumps
        if len(self.regions) < 20:
            self.avgscore /= 3
            self.maxerror = 100

    def addRegionIfClose(self, r, dist):
        '''Only add the region to this clump if it's within a certain
        distance of the avg centrepoint'''
        self.calccentrepoint()
        (lat,lon) = self.centreloc
        (la, lo) = r.latlon
        if dist > cuav_util.gps_distance(lat, lon, la, lo):
            self.regions.append((la, lo, r.score))
            return True
        else:
            return False


class LandingZone:
    def __init__(self):
        self.landingzone = None
        self.landingzonemaxrange = None
        self.landingzonemaxscore = None
        self.regionClumps = []

    def checkaddregion(self, r):
        '''Add a region to the list of landing zone clumps, else make a new
        LZ clump
        '''

        #if this is the first region, make a clump
        if len(self.regionClumps) == 0:
            rl = RegionClump(r)
            self.regionClumps.append(rl)
        else:
            #add to first clump within 5m
            for cl in self.regionClumps:
                if cl.addRegionIfClose(r, 5):
                    return
            #else make a new clump, as we couldn't find a good clump for it
            rl = RegionClump(r)
            self.regionClumps.append(rl)

    def calclandingzone(self):
        '''Given the current regionclumps, find the landing zone and
        confidence level. Returns True is a zone was found, False otherwise
        '''
        #1. Clump all images that are within 15m of each other (done in checkaddregion)

        #2. Get the average score
        # Done in calclandingzone

        #3. Get the clump with the highest median (clump must have >15 images in it)
        bestlz = None
        for cl in self.regionClumps:
            #print("Clump has %u locations with avg score %u" % (len(cl.regions), cl.avgscore))
            if bestlz is None or (bestlz.avgscore < cl.avgscore and len(cl.regions) > 15):
                bestlz = cl

        #4. And store
        if bestlz and len(bestlz.regions) > 15:
            self.landingzone = bestlz.centreloc
            self.landingzonemaxrange = bestlz.maxerror
            self.landingzonemaxscore = bestlz.avgscore
            return True
        else:
            self.landingzone = None
            self.landingzonemaxrange = None
            self.landingzonemaxscore = None
            return False

