#!/usr/bin/python
'''
Outback Challenge Mission Generator
It read in a KML file and generate mission waypoints (ie. search pattern)
Created by Stephen Dade (stephen_dade@hotmail.com)
'''

import numpy, os, time, sys, xml.dom.minidom, math, numpy
import argparse

from cuav.lib import cuav_util
from pymavlink import mavwp, mavutil
from MAVProxy.modules.mavproxy_map import mp_slipmap

class MissionGenerator():
    '''Mission Generator Class'''

    def __init__(self, inFile='obc.kml', altitude=90, cmac=True):
        self.inFile = inFile
        self.dom = xml.dom.minidom.parse(inFile)

        self.searchArea = []
        self.missionBounds = []
        self.entryPoints = []
        self.exitPoints = []
        self.SearchPattern = []
        if cmac:
            self.joePos = (-35.362748, 149.162257, 80)
            self.landingHeading = 353.0
            self.landingPt = (-35.362879, 149.165190)
        else:
            self.joePos = (-26.624864, 151.848349, 80)
            self.landingHeading = 349.1
            self.landingPt = (-26.585107, 151.840798)

        # height to loiter at airfield home
        self.airportHeight = 100
        self.MAVpointLoader = None            

    def parsePlacemark(self, point):
        '''parse a Placemark XML object'''
        name = self.getElement(point.getElementsByTagName('name')[0])
        pt = point.getElementsByTagName('Point')
        coords = pt[0].getElementsByTagName('coordinates')[0]
        ctext = self.getElement(coords)
        (lon, lat, alt) = ctext.split(',')
        lat = float(lat)
        lon = float(lon)
        return (name, lat, lon)

    def Process(self, searchMask="SA-", missionBoundaryMask="MB-"):
        '''Processes the imported xml file for points'''
        
        self.searchArea = []
        self.missionBounds = []
        #get a list of all points in the kml file:
        airf = self.dom.getElementsByTagName('Placemark')
        for point in airf:
            (name, lat, lon) = self.parsePlacemark(point)
            if name == "Airfield Home":
                self.airfieldHome = (lat, lon)
            if name.startswith(searchMask):
                self.searchArea.append((lat, lon))
            if name.startswith(missionBoundaryMask):
                self.missionBounds.append((lat, lon))

        #print "Search Area = " + str(self.searchArea)
        #print "Mission Boundary = " + str(self.missionBounds)

        #get the bounding box of the search area
        self.boundingBoxLat = [(self.searchArea[0])[0], (self.searchArea[0])[0]]
        self.boundingBoxLong = [(self.searchArea[0])[1], (self.searchArea[0])[1]]
        for point in self.searchArea:
            if point[0] < self.boundingBoxLat[0]:
                self.boundingBoxLat[0] = point[0]
            if point[0] > self.boundingBoxLat[1]:
                self.boundingBoxLat[1] = point[0]       
            if point[1] < self.boundingBoxLong[0]:
                self.boundingBoxLong[0] = point[1]
            if point[1] > self.boundingBoxLong[1]:
                self.boundingBoxLong[1] = point[1]  

        self.boundingBox = [(self.boundingBoxLat[0], self.boundingBoxLong[0]), (self.boundingBoxLat[1], self.boundingBoxLong[0]), (self.boundingBoxLat[0], self.boundingBoxLong[1]), (self.boundingBoxLat[1], self.boundingBoxLong[1])] 
        #print "Bounding box is: " + str(self.boundingBoxLat) + " ... " + str(self.boundingBoxLong)

        #for point in self.searchArea:
        #    print self.getDistAndBearing(self.airfieldHome, point)

    def CreateEntryExitPoints(self, entryLane, exitLane, alt=100):
        '''Create the waypoints for the entry and exit waypoints to fly
        before/after the search pattern. alt is the altitude (relative 
        to ground) of the points'''

        self.entryPoints = []
        self.exitPoints = []
        listentry = entryLane.split(',')
        listexit = exitLane.split(',')
        #print "here" + str(listentry)

        self.airfieldHome = (self.airfieldHome[0], self.airfieldHome[1], alt)

        airf = self.dom.getElementsByTagName('Placemark')
        for entPoint in listentry:
            for point in airf:
                (name, lat, lon) = self.parsePlacemark(point)
                if name == entPoint:
                    self.entryPoints.append((lat, lon, alt))
                    print "Entry - " + str(self.entryPoints[-1])

        for exPoint in listexit:
            for point in airf:
                (name, lat, lon) = self.parsePlacemark(point)
                if name == exPoint:
                    self.exitPoints.append((lat, lon, alt))
                    print "Exit - " + str(self.exitPoints[-1])

    def CreateSearchPattern(self, width=50.0, overlap=10.0, offset=10, wobble=10, alt=100):
        '''Generate the waypoints for the search pattern, using alternating strips
        width is the width (m) of each strip, overlap is the % overlap between strips, 
        alt is the altitude (relative to ground) of the points'''
        self.SearchPattern = []

        #find the nearest point to Airfield Home - use this as a starting point (if entry lanes are not used)
        if len(self.entryPoints) == 0:
            nearestdist = cuav_util.gps_distance(self.airfieldHome[0], self.airfieldHome[1], self.searchArea[0][0], self.searchArea[0][1])
            nearest = self.searchArea[0]
            for point in self.searchArea:
                newdist = cuav_util.gps_distance(self.airfieldHome[0], self.airfieldHome[1], point[0], point[1])
                if newdist < nearestdist:
                    nearest = point
                    nearestdist = newdist
        else:
            nearestdist = cuav_util.gps_distance(self.entryPoints[0][0], self.entryPoints[0][1], self.searchArea[0][0], self.searchArea[0][1])
            nearest = self.searchArea[0]
            for point in self.searchArea:
                newdist = cuav_util.gps_distance(self.entryPoints[0][0], self.entryPoints[0][1], point[0], point[1])
                #print "dist = " + str(newdist)
                if newdist < nearestdist:
                    nearest = point
                    nearestdist = newdist

        #print "Start = " + str(nearest) + ", dist = " + str(nearestdist)

        #the search pattern will run between the longest side from nearest
        bearing1 = cuav_util.gps_bearing(nearest[0], nearest[1], self.searchArea[self.searchArea.index(nearest)-1][0], self.searchArea[self.searchArea.index(nearest)-1][1])
        bearing2 = cuav_util.gps_bearing(nearest[0], nearest[1], self.searchArea[self.searchArea.index(nearest)+1][0], self.searchArea[self.searchArea.index(nearest)+1][1])
        dist1 = cuav_util.gps_distance(nearest[0], nearest[1], self.searchArea[self.searchArea.index(nearest)-1][0], self.searchArea[self.searchArea.index(nearest)-1][1])
        dist2 = cuav_util.gps_distance(nearest[0], nearest[1], self.searchArea[self.searchArea.index(nearest)+1][0], self.searchArea[self.searchArea.index(nearest)+1][1])
        if dist1 > dist2:
            self.searchBearing = bearing1
        else:
            self.searchBearing = bearing2

        #the search pattern will then run parallel between the two furthest points in the list
        #searchLine = (0, 0)
        #for point in self.searchArea: 
        #    newdist = cuav_util.gps_distance(point[0], point[1], self.searchArea[self.searchArea.index(point)-1][0], self.searchArea[self.searchArea.index(point)-1][1])
        #    if newdist > searchLine[0]:
        #        searchLine = (newdist, cuav_util.gps_bearing(point[0], point[1], self.searchArea[self.searchArea.index(point)-1][0], self.searchArea[self.searchArea.index(point)-1][1]))

        #self.searchBearing = searchLine[1]
        

        #need to find the 90 degree bearing to searchBearing that is inside the search area. This
        #will be the bearing we increment the search rows by
        #need to get the right signs for the bearings, depending which quadrant the search area is in wrt nearest
        if not cuav_util.polygon_outside(cuav_util.gps_newpos(nearest[0], nearest[1], (self.searchBearing + 45) % 360, 10), self.searchArea):
            self.crossBearing = (self.searchBearing + 90) % 360
        elif not cuav_util.polygon_outside(cuav_util.gps_newpos(nearest[0], nearest[1], (self.searchBearing + 135) % 360, 10), self.searchArea):
            self.crossBearing = (self.searchBearing + 90) % 360
            self.searchBearing = (self.searchBearing + 180) % 360
        elif not cuav_util.polygon_outside(cuav_util.gps_newpos(nearest[0], nearest[1], (self.searchBearing - 45) % 360, 10), self.searchArea):
            self.crossBearing = (self.searchBearing - 90) % 360
        else:
            self.crossBearing = (self.searchBearing - 90) % 360
            self.searchBearing = (self.searchBearing - 180) % 360

        print "Search bearing is " + str(self.searchBearing) + "/" + str((self.searchBearing + 180) % 360)
        print "Cross bearing is: " + str(self.crossBearing)

        #the distance between runs is this:
        self.deltaRowDist = width - width*(float(overlap)/100)
        if self.deltaRowDist <= 0:
            print "Error, overlap % is too high"
            return
        print "Delta row = " + str(self.deltaRowDist)

        #expand the search area to 1/2 deltaRowDist to ensure full coverage

        #we are starting at the "nearest" and mowing the lawn parallel to "self.searchBearing"
        #first waypoint is right near the Search Area boundary (without being on it) (10% of deltaRowDist
        #on an opposite bearing (so behind the search area)
        nextWaypoint =  cuav_util.gps_newpos(nearest[0], nearest[1], self.crossBearing, self.deltaRowDist/10)
        print "First = " + str(nextWaypoint)
        #self.SearchPattern.append(firstWaypoint)

        #mow the lawn, every 2nd row:
        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #print "Projecting " + str(nextWaypoint) + " along " + str(self.searchBearing)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextW[0], nextW[1]) < cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextnextW[0], nextnextW[1]):
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], (self.searchBearing + 180) % 360, (offset+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, (offset+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.crossBearing, self.deltaRowDist*2)
                self.searchBearing = (self.searchBearing + 180) % 360
            else:
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], (self.searchBearing + 180) % 360, offset+wobble))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, (offset+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextW[0], nextW[1], self.crossBearing, self.deltaRowDist*2)
                self.searchBearing = (self.searchBearing + 180) % 360

            print "Next = " + str(nextWaypoint)
        
        #go back and do the rows we missed. There still might be one more row to do in 
        # the crossbearing direction, so check for that first
        nextWaypoint = cuav_util.gps_newpos(nextWaypoint[0], nextWaypoint[1], self.crossBearing, -self.deltaRowDist)
        pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
        if pts == 0:
            nextWaypoint = cuav_util.gps_newpos(nextWaypoint[0], nextWaypoint[1], self.crossBearing, -2*self.deltaRowDist)
            self.crossBearing = (self.crossBearing + 180) % 360
        else:
            self.crossBearing = (self.crossBearing + 180) % 360

        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #print "Projecting " + str(nextWaypoint) + " along " + str(self.searchBearing)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextW[0], nextW[1]) < cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextnextW[0], nextnextW[1]):
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], (self.searchBearing + 180) % 360, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.crossBearing, self.deltaRowDist*2)
                self.searchBearing = (self.searchBearing + 180) % 360
            else:
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], (self.searchBearing + 180) % 360, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextW[0], nextW[1], self.crossBearing, self.deltaRowDist*2)
                self.searchBearing = (self.searchBearing + 180) % 360

            print "Next(alt) = " + str(nextWaypoint)

        #add in the altitude points (relative to airfield home)
        for point in self.SearchPattern:
            self.SearchPattern[self.SearchPattern.index(point)] = (point[0], point[1], alt)

    def isOutsideSearchAreaBoundingBox(self, lat, longy):
        '''Checks if the long/lat pair is inside the search area
        bounding box. Returns true if it is inside'''

        if cuav_util.polygon_outside((lat, longy), self.boundingBox):
            return 1
        else:
            return 0

    def ExportSearchPattern(self, filename='search.kml'):
        '''Exports the current search pattern to Google Earth kml file'''

        if len(self.SearchPattern) == 0:
            return

        #and actually write the file
        with open(filename, 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\" xmlns:gx=\"http://www.google.com/kml/ext/2.2\" xmlns:kml=\"http://www.opengis.net/kml/2.2\">\n")
            f.write("<Document>\n")
            f.write("    <name>" + filename + "</name>\n")

            #loop around the waypoints
            for point in self.SearchPattern:
                f.write("    <Placemark>\n")
                f.write("        <name>SP-" + str(self.SearchPattern.index(point)) + "</name>\n")
                f.write("        <Point>\n")
                f.write("            <altitudeMode>absolute</altitudeMode>\n")
                f.write("            <coordinates>" + str(point[1]) + "," + str(point[0]) + "," + str(point[2]) + "</coordinates>\n")
                f.write("        </Point>\n")
                f.write("    </Placemark>\n")

            #make a polygon for easier viewing
            f.write("    <Placemark>\n")
            f.write("        <name>Search Pattern</name>\n")
            f.write("        <LineString>\n")
            f.write("            <tessellate>0</tessellate>\n")
            f.write("            <altitudeMode>absolute</altitudeMode>\n")
            f.write("            <coordinates>\n")
            for point in self.SearchPattern:
                f.write("                " + str(point[1]) + "," + str(point[0]) + "," + str(point[2]) + "\n")
            f.write("            </coordinates>\n")
            f.write("        </LineString>\n")
            f.write("    </Placemark>\n")
            
            f.write("</Document>\n")
            f.write("</kml>")


    def getElement(self, element):
        '''Get and XML element'''
        return self.getText(element.childNodes)


    def getText(self, nodelist):
        '''Get the text inside an XML node'''
        rc = ""
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.nodeValue
        return rc

    def projectBearing(self, bearing, startPos, searchArea):
        '''Projects bearing from startPos until it reaches the edge(s) of
        searchArea (list of lat/long tuples. Returns the First/Last position(s)'''

        coPoints = []

        #for each line in the search are border, get any overlaps with startPos on bearing(+-180)
        for point in searchArea:
            dist = cuav_util.gps_distance(point[0], point[1], searchArea[searchArea.index(point)-1][0], searchArea[searchArea.index(point)-1][1])
            theta2 = cuav_util.gps_bearing(point[0], point[1], searchArea[searchArea.index(point)-1][0], searchArea[searchArea.index(point)-1][1])
            posn = self.Intersection(startPos, bearing, point, theta2)
            if posn != 0 and cuav_util.gps_distance(posn[0], posn[1], point[0], point[1]) < dist:
                coPoints.append(posn)
            posn = self.Intersection(startPos, (bearing + 180) % 360, point, theta2)
            if posn != 0 and cuav_util.gps_distance(posn[0], posn[1], point[0], point[1]) < dist:
                coPoints.append(posn)

        #if there's more than two points in coPoints, return the furthest away points
        if len(coPoints) < 2:
            #print str(len(coPoints)) + " point i-sect"
            return 0
        elif len(coPoints) == 2:
            return coPoints
        else:
            dist = 0
            for point in coPoints:
                for pt in coPoints:
                    if cuav_util.gps_distance(pt[0], pt[1], point[0], point[1]) > dist:
                        dist = cuav_util.gps_distance(pt[0], pt[1], point[0], point[1])
                        newcoPoints = [point, pt]
            return newcoPoints


    def Intersection(self, pos1, theta1, pos2, theta2):
        '''This will find the intersection between two positions and bearings
           it will return the intersection lat/long, or 0 if no intersection'''
        brng1 = theta1
        brng2 = theta2
        lat1 = numpy.radians(pos1[0])
        lon1 = numpy.radians(pos1[1])
        lat2 = numpy.radians(pos2[0])
        lon2 = numpy.radians(pos2[1])
        brng13 = numpy.radians(brng1)
        brng23 = numpy.radians(brng2)
        dLat = lat2-lat1
        dLon = lon2-lon1
  
        dist12 = 2*numpy.arcsin( numpy.sqrt( numpy.sin(dLat/2)*numpy.sin(dLat/2) + numpy.cos(lat1)*numpy.cos(lat2)*numpy.sin(dLon/2)*numpy.sin(dLon/2) ) )
        if dist12 == 0:
            return 0
  
        #initial/final bearings between points
        brngA = numpy.arccos( ( numpy.sin(lat2) - numpy.sin(lat1)*numpy.cos(dist12) ) / ( numpy.sin(dist12)*numpy.cos(lat1) ) )
        #if (isNaN(brngA)) brngA = 0;  // protect against rounding
        brngB = numpy.arccos( ( numpy.sin(lat1) - numpy.sin(lat2)*numpy.cos(dist12) ) / ( numpy.sin(dist12)*numpy.cos(lat2) ) )
  
        if numpy.sin(lon2-lon1) > 0:
            brng12 = brngA
            brng21 = 2*numpy.pi - brngB
        else:
           brng12 = 2*numpy.pi - brngA
           brng21 = brngB
  
        alpha1 = (brng13 - brng12 + numpy.pi) % (2*numpy.pi) - numpy.pi  # angle 2-1-3
        alpha2 = (brng21 - brng23 + numpy.pi) % (2*numpy.pi) - numpy.pi  # angle 1-2-3
  
        if numpy.sin(alpha1)==0 and numpy.sin(alpha2)==0:
            return 0  # infinite intersections
        if numpy.sin(alpha1)*numpy.sin(alpha2) < 0:
            return 0      # ambiguous intersection
   
        alpha3 = numpy.arccos( -numpy.cos(alpha1)*numpy.cos(alpha2) + numpy.sin(alpha1)*numpy.sin(alpha2)*numpy.cos(dist12) )
        dist13 = numpy.arctan2( numpy.sin(dist12)*numpy.sin(alpha1)*numpy.sin(alpha2), numpy.cos(alpha2)+numpy.cos(alpha1)*numpy.cos(alpha3) )
        lat3 = numpy.arcsin( numpy.sin(lat1)*numpy.cos(dist13) + numpy.cos(lat1)*numpy.sin(dist13)*numpy.cos(brng13) )
        dLon13 = numpy.arctan2( numpy.sin(brng13)*numpy.sin(dist13)*numpy.cos(lat1), numpy.cos(dist13)-numpy.sin(lat1)*numpy.sin(lat3) )
        lon3 = lon1+dLon13
        lon3 = (lon3+3*numpy.pi) % (2*numpy.pi) - numpy.pi  # normalise to -180..+180
  
        return (numpy.degrees(lat3), numpy.degrees(lon3));

    def getMapPolygon(self, loiterInSearchArea=1):
        '''Returns a mp_Slipmap compatible (2d) polygon'''
        meanPoint = tuple(numpy.average(self.SearchPattern, axis=0))
        #print "Mean = " + str(meanPoint)

        if loiterInSearchArea == 1:
            tmp = self.entryPoints[:-1] + self.SearchPattern + self.exitPoints
        else:
            tmp = self.entryPoints[:-1] + self.SearchPattern + self.exitPoints

        return [(row[0], row[1]) for row in tmp]

    def getPolygonLength(self):
        '''Returns the search pattern path length (metres)'''
        distance = 0
        totPoint = self.entryPoints + self.SearchPattern + self.exitPoints

        for point in totPoint:
            if point != totPoint[-1]:
                distance = distance + cuav_util.gps_distance(point[0], point[1], totPoint[totPoint.index(point)-1][0], totPoint[totPoint.index(point)-1][1])

        return int(distance)

    def waypoint(self, lat, lon, alt,
                 frame=mavutil.mavlink.MAV_FRAME_GLOBAL_TERRAIN_ALT,
                 cmd=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                 param=[0,0,0,0]):
        '''return a navigation waypoint'''
        return mavutil.mavlink.MAVLink_mission_item_message(self.MAVpointLoader.target_system,
                                                            self.MAVpointLoader.target_component, 0,
                                                            frame, cmd, 1, 1, *param, x=lat, y=lon, z=int(alt))

    def jump(self, wpnum):
        '''return a jump waypoint'''
        return self.waypoint(0,0,0, cmd=mavutil.mavlink.MAV_CMD_DO_JUMP, param=[wpnum, -1, 0, 0])

    def command(self, cmd, param=[0,0,0,0]):
        '''return a jump waypoint'''
        return self.waypoint(0,0,0, cmd=cmd, param=param)
                     

    def exportToMAVProxy(self, loiterInSearchArea=1, basealt=100, outname="way.txt"):
        '''Exports the airfield home, entry points, search pattern and exit points to MAVProxy'''

        #make a fake waypoint loader for testing purposes, if we're not
        #running within MAVProxy
        if self.MAVpointLoader is None:
            print "No loader - creating one"
            self.MAVpointLoader = mavwp.MAVWPLoader()
        MAVpointLoader = self.MAVpointLoader

        entryjump = []
        exitjump = []

        #clear out all the old waypoints
        MAVpointLoader.clear()

        TargetSys = MAVpointLoader.target_system
        TargetComp = MAVpointLoader.target_component

        #WP0 - add landingPt as waypoint 0. This gets replaced when GPS gets lock
        w = self.waypoint(*self.landingPt, alt=basealt,
                          frame=mavutil.mavlink.MAV_FRAME_GLOBAL)
        MAVpointLoader.add(w, comment='Airfield home')

        #WP2 - takeoff, then jump to entry lanes
        w = self.waypoint(*cuav_util.gps_newpos(*self.landingPt, bearing=self.landingHeading+175, distance=300), alt=40,
                          cmd=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, param=[12, 0, 0, 0])
        MAVpointLoader.add(w, comment="Takeoff")
        entryjump.append(MAVpointLoader.count())
        w = self.jump(0)
        MAVpointLoader.add(w, comment='Jump to entry lane')

        # landing approach 1
        landing_approach_wpnum = MAVpointLoader.count()
        w = self.waypoint(*cuav_util.gps_newpos(*self.landingPt, bearing=self.landingHeading+180, distance=1000), alt=80)
        MAVpointLoader.add(w, comment='Landing approach')

        # drop our speed
        w = self.command(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, param=[0, 23, 20, 0])
        MAVpointLoader.add(w, comment='Change to 23 m/s')
        w = self.command(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, param=[1, 23, 20, 0])
        MAVpointLoader.add(w, comment='Change throttle to 20%%')

        # landing approach 1
        landing_approach_wpnum = MAVpointLoader.count()
        w = self.waypoint(*cuav_util.gps_newpos(*self.landingPt, bearing=self.landingHeading+180, distance=500), alt=50)
        MAVpointLoader.add(w, comment='Landing approach 2')

        # landing
        w = self.waypoint(self.landingPt[0], self.landingPt[1], 0, cmd=mavutil.mavlink.MAV_CMD_NAV_LAND)
        MAVpointLoader.add(w, comment='Landing')

        # comms Failure. Loiter at EL-4 for 2 minutes then fly to airfield home and loiter
        point = self.exitPoints[1]
        w = self.waypoint(point[0], point[1], self.airportHeight)
        MAVpointLoader.add(w, comment='Comms Failure')

        w = self.command(mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME, param=[120, 0, 0, 0])
        MAVpointLoader.add(w, comment='loiter 2 minutes')

        w = self.waypoint(self.airfieldHome[0], self.airfieldHome[1], self.airportHeight)
        MAVpointLoader.add(w, comment='Airfield home')

        w = self.command(mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM)
        MAVpointLoader.add(w, comment='loiter')


        # GPS failure. Loiter in place for 30s then direct to airfield home
        w = self.command(mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME, param=[30,0,0,0])
        MAVpointLoader.add(w, comment='GPS fail - loiter 30 secs')

        w = self.waypoint(self.airfieldHome[0], self.airfieldHome[1], self.airportHeight)
        MAVpointLoader.add(w, comment='Airfield home')

        w = self.command(mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM)
        MAVpointLoader.add(w, comment='loiter')

        dropalt = self.joePos[2]
        joe = (self.joePos[0], self.joePos[1])

        crosswp = MAVpointLoader.count()
        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=60, distance=250), alt=dropalt)        
        MAVpointLoader.add(w, comment='cross 1')

        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=240, distance=250), alt=dropalt)        
        MAVpointLoader.add(w, comment='cross 2')

        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=300, distance=250), alt=dropalt)        
        MAVpointLoader.add(w, comment='cross 3')

        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=120, distance=250), alt=dropalt)        
        MAVpointLoader.add(w, comment='cross 4')

        w = self.jump(crosswp)
        MAVpointLoader.add(w, comment='Jump to cross')
        
        # joe approach point
        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=0, distance=350), alt=dropalt)    
        MAVpointLoader.add(w, comment='Joe approach')

        w = self.command(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, param=[0, 20, 50, 0])
        MAVpointLoader.add(w, comment='Change to 20 m/s')

        # joe location. We use a default acceptance radius of 35 meters. Will be adjusted with 'wp param'
        # command
        w = self.waypoint(*joe, alt=dropalt, param=[0, 70, 0, 0])  
        MAVpointLoader.add(w, comment='Joe Location')
        
        w = self.command(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, [8, 1500, 0, 0])
        MAVpointLoader.add(w, comment='Drop bottle 1')

        w = self.command(mavutil.mavlink.MAV_CMD_CONDITION_DELAY, [2.0, 0, 0, 0])
        MAVpointLoader.add(w, comment='Drop delay')

        w = self.command(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, [6, 1050, 0, 0])
        MAVpointLoader.add(w, comment='Drop bottle 2')

        w = self.command(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, param=[0, 28, 50, 0])
        MAVpointLoader.add(w, comment='Change to 28 m/s')

        w = self.waypoint(*cuav_util.gps_newpos(*joe, bearing=180, distance=250), alt=dropalt)        
        MAVpointLoader.add(w, comment='Joe after')

        w = self.jump(crosswp)
        MAVpointLoader.add(w, comment='Jump to cross')

        #print "Done AF home"
        #WP12 - WPn - and add in the rest of the waypoints - Entry lane, search area, exit lane
        entry_wpnum = MAVpointLoader.count()
        for i in range(1):
            point = self.entryPoints[i]
            w = self.waypoint(point[0], point[1], self.airportHeight)
            MAVpointLoader.add(w, comment='Entry %u' % (i+1))
        endentry_wpnum = MAVpointLoader.count()
        w = self.jump(0)
        MAVpointLoader.add(w, comment='Jump to search mission')

        # exit points
        exit_wpnum = MAVpointLoader.count()
        for i in range(len(self.exitPoints)):
            point = self.exitPoints[i]
            w = self.waypoint(point[0], point[1], self.airportHeight)
            MAVpointLoader.add(w, comment='Exit point %u' % (i+1))

        w = self.jump(landing_approach_wpnum)
        MAVpointLoader.add(w, comment='Jump to landing approach')

        # search pattern
        MAVpointLoader.wp(endentry_wpnum).param1 = MAVpointLoader.count()
        for i in range(len(self.SearchPattern)):
            point = self.SearchPattern[i]
            w = self.waypoint(point[0], point[1], point[2])
            MAVpointLoader.add(w, comment='Search %u' % (i+1))

        #if desired, loiter in the search area for a bit
        if loiterInSearchArea == 1:
            meanPoint = tuple(numpy.average(self.SearchPattern, axis=0))
            w = self.waypoint(meanPoint[0], meanPoint[1], meanPoint[2],
                              cmd=mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
                              param=[600, 0, 0, 0])
            MAVpointLoader.add(w, comment='Loiter in search area for 10 minutes')

        exitjump.append(MAVpointLoader.count())        
        w = self.jump(0)
        MAVpointLoader.add(w, comment='Jump to exit lane')

        # fixup jump waypoint numbers
        for wnum in entryjump:
            MAVpointLoader.wp(wnum).param1 = entry_wpnum
        for wnum in exitjump:
            MAVpointLoader.wp(wnum).param1 = exit_wpnum

        #export the waypoints to a MAVLink compatible format/file
        waytxt = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                              'data', outname)
        MAVpointLoader.save(waytxt)
        print "Waypoints exported to %s" % waytxt

        # create fence.txt
        fenceloader = mavwp.MAVFenceLoader()
        fp = mavutil.mavlink.MAVLink_fence_point_message(0, 0, 0, 0, self.airfieldHome[0], self.airfieldHome[1])
        fenceloader.add(fp)
        for p in gen.missionBounds:
            fp = mavutil.mavlink.MAVLink_fence_point_message(0, 0, 0, 0, float(p[0]), float(p[1]))
            fenceloader.add(fp)
        # close the polygon
        p = gen.missionBounds[0]
        fp = mavutil.mavlink.MAVLink_fence_point_message(0, 0, 0, 0, float(p[0]), float(p[1]))
        fenceloader.add(fp)
        fencetxt = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', "fence.txt")
        fenceloader.save(fencetxt)
        print "Fence exported to %s" % fencetxt


        #print strMAV

    def getCameraWidth(self, alt, mpp100, camerares):
        '''Using the camera parameters, with the width of the
        ground strip that the camera can see from a particular altitude'''
        return camerares * mpp100 * alt / 100.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a OBC mission, based on kml file")
    parser.add_argument("file", default='..//data//OBC Waypoints 2014.kml', help="input file")
    parser.add_argument("--searchAreaMask", default='SA-', help="name mask of search area waypoints")
    parser.add_argument("--missionBoundaryMask", default='MB-', help="name mask of mission boundary waypoints")
    parser.add_argument("--searchAreaOffset", type=int, default=150, help="distance waypoints will be placed outside search area")
    parser.add_argument("--wobble", type=int, default=10, help="Make every second row slightly offset. Aids in viewing the overlaps")
    parser.add_argument("--width", type=int, default=0, help="Width (m) of each scan row. 0 to use camera params")
    parser.add_argument("--overlap", type=int, default=60, help="% overlap between rows")
    parser.add_argument("--entryLane", default='EL-01,EL-02', help="csv list of waypoints before search")
    parser.add_argument("--exitLane", default='EL-03,EL-04', help="csv list of waypoints after search")
    parser.add_argument("--altitude", type=int, default=100, help="Altitude of waypoints")
    parser.add_argument("--loiterInSearchArea", type=int, default=1, help="1 if UAV loiters in search area at end of search. 0 if it goes home")
    parser.add_argument("--sutton", action='store_true', default=False, help="use sutton WP")
    parser.add_argument("--cmac", action='store_true', default=False, help="use CMAC WP")
    parser.add_argument("--outname", default='way.txt', help="name in data dir")
    parser.add_argument("--basealt", default=0, type=int, help="base altitude")
    parser.add_argument("--mpp100", default=0.098, type=float, help="camera meters per pixel at 100m")
    parser.add_argument("--camerares", default=1280, type=float, help="Camera resolution")
    parser.add_argument("--map", help="camera meters per pixel at 100m")
    

    args = parser.parse_args()

    gen = MissionGenerator(args.file, args.cmac)
    gen.Process(args.searchAreaMask, args.missionBoundaryMask)
    gen.CreateEntryExitPoints(args.entryLane, args.exitLane)

    groundWidth = args.width
    #are we using the camera params to get the size of each search strip?
    if args.width == 0:
        groundWidth = gen.getCameraWidth(args.altitude, args.mpp100, args.camerares)
    print "Strip width = " + str(groundWidth)

    gen.CreateSearchPattern(width = groundWidth, overlap=args.overlap, offset=args.searchAreaOffset, wobble=args.wobble, alt=args.altitude)
    
    gen.ExportSearchPattern()

    #start a map
    sm = mp_slipmap.MPSlipMap(lat=gen.getMapPolygon(loiterInSearchArea=args.loiterInSearchArea)[0][0], lon=gen.getMapPolygon(loiterInSearchArea=args.loiterInSearchArea)[0][1], elevation=True, service='GoogleSat')
    sm.add_object(mp_slipmap.SlipPolygon('Search Pattern', gen.getMapPolygon(), layer=1, linewidth=2, colour=(0,255,0)))

    #get the search pattern distance
    print "Total Distance = " + str(gen.getPolygonLength()) + "m"

    #and export to MAVProxy
    gen.exportToMAVProxy(args.loiterInSearchArea, args.basealt, args.outname)

    #and to google earth
    gen.ExportSearchPattern()
