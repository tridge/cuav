#!/usr/bin/python
'''
Outback Challenge Mission Generator
It read in a KML file and generate mission waypoints (ie. search pattern)
Created by Stephen Dade (stephen_dade@hotmail.com)
'''

import numpy, os, time, sys, xml.dom.minidom, math, numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'MAVProxy', 'modules', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'mavlink', 'pymavlink'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'camera'))
import mp_slipmap, mp_util, cuav_util, mp_elevation, mavwp, mavutil, mavlinkv10
from cam_params import CameraParams
from mavlinkv10 import *

class MissionGenerator():
    '''Mission Generator Class'''

    def __init__(self, inFile='obc.kml'):
        self.inFile = inFile
        self.dom = xml.dom.minidom.parse(inFile)

        self.searchArea = []
        self.missionBounds = []
        self.entryPoints = []
        self.exitPoints = []
        self.SearchPattern = []
        self.joeApproach = (-26.623860, 151.847557, 150)
        self.joeDrop = (-26.624864, 151.848349, 150)

    def Process(self, searchMask="SA-", missionBoundaryMask="MB-"):
        '''Processes the imported xml file for points'''
        
        self.searchArea = []
        self.missionBounds = []
        #get a list of all points in the kml file:
        airf = self.dom.getElementsByTagName('Placemark')
        for point in airf:
            if self.getElement(point.getElementsByTagName('name')[0]) == "Airfield Home":
                self.airfieldHome = (float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0])))
                #print "Airfield Home = " + str(self.airfieldHome)
            if searchMask in self.getElement(point.getElementsByTagName('name')[0]):
                self.searchArea.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))
            if missionBoundaryMask in self.getElement(point.getElementsByTagName('name')[0]):
                self.missionBounds.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))

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
        for point in airf:
            if self.getElement(point.getElementsByTagName('name')[0]) in listentry:
                self.entryPoints.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0])), alt))
                #print "Entry - " + str(self.entryPoints[-1])

        for point in airf:
            if self.getElement(point.getElementsByTagName('name')[0]) in listexit:
                self.exitPoints.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0])), alt))
                #print "Exit - " + str(self.exitPoints[-1])

    def CreateSearchPattern(self, width=50.0, overlap=10.0, offset=10, wobble=1, alt=100):
        '''Generate the waypoints for the search pattern, using alternating strips
        width is the width (m) of each strip, overlap is the % overlap between strips, 
        alt is the altitude (relative to ground) of the points'''
        self.SearchPattern = []

        #find the nearest point to Airfield Home - use this as a starting point
        nearestdist = cuav_util.gps_distance(self.airfieldHome[0], self.airfieldHome[1], self.searchArea[0][0], self.searchArea[0][1])
        nearest = self.searchArea[0]
        for point in self.searchArea:
            newdist = cuav_util.gps_distance(self.airfieldHome[0], self.airfieldHome[1], point[0], point[1])
            if newdist < nearestdist:
                nearest = point
                nearestdist = newdist

        #print "Start = " + str(nearest) + ", dist = " + str(nearestdist)

        #the search pattern will then run parallel between the two furthest points in the list
        searchLine = (0, 0)
        for point in self.searchArea: 
            newdist = cuav_util.gps_distance(point[0], point[1], self.searchArea[self.searchArea.index(point)-1][0], self.searchArea[self.searchArea.index(point)-1][1])
            if newdist > searchLine[0]:
                searchLine = (newdist, cuav_util.gps_bearing(point[0], point[1], self.searchArea[self.searchArea.index(point)-1][0], self.searchArea[self.searchArea.index(point)-1][1]))

        self.searchBearing = searchLine[1]
        #print "Search bearing is " + str(self.searchBearing) + "/" + str((self.searchBearing + 180) % 360)

        #need to find the 90 degree bearing to searchBearing that is inside the search area. This
        #will be the bearing we increment the search rows by
        if not cuav_util.polygon_outside(cuav_util.gps_newpos(nearest[0], nearest[1], (self.searchBearing + 45) % 360, 10), self.searchArea):
            self.crossBearing = (self.searchBearing + 90) % 360
        else:
            self.crossBearing = (self.searchBearing - 90) % 360
        #print "Cross bearing is: " + str(self.crossBearing)

        #the distance between runs is this:
        self.deltaRowDist = width - width*(float(overlap)/100)
        if self.deltaRowDist <= 0:
            print "Error, overlap % is too high"
            return
        #print "Delta row = " + str(self.deltaRowDist)

        #we are starting at the "nearest" and mowing the lawn parallel to "self.searchBearing"
        #first waypoint is width*(overlap/100) on an opposite bearing (so behind the search area)
        nextWaypoint =  cuav_util.gps_newpos(nearest[0], nearest[1], self.crossBearing, self.deltaRowDist/2)
        print "First = " + str(nextWaypoint)
        #self.SearchPattern.append(firstWaypoint)

        #mow the lawn, every 2nd row:
        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextW[0], nextW[1]) < cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextnextW[0], nextnextW[1]):
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, -offset*(1+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, offset*(1+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.crossBearing, self.deltaRowDist*2)
            else:
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, offset*(1+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, -offset*(1+wobble)))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextW[0], nextW[1], self.crossBearing, self.deltaRowDist*2)

            #print "Next = " + str(nextWaypoint)
        
        #go back and do the rows we missed. There still might be one more row to do in 
        # the crossbearing direction, so check for that first
        nextWaypoint = cuav_util.gps_newpos(nextWaypoint[0], nextWaypoint[1], self.crossBearing, self.deltaRowDist)
        pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
        if pts == 0:
            nextWaypoint = cuav_util.gps_newpos(self.SearchPattern[-1][0], self.SearchPattern[-1][1], self.crossBearing, self.deltaRowDist)
            self.crossBearing = (self.crossBearing + 180) % 360
        else:
            self.crossBearing = (self.crossBearing + 180) % 360

        while True:
            pts = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            #check if we're outside the search area
            if pts == 0:
                break
            (nextW, nextnextW) = (pts[0], pts[1])
            if cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextW[0], nextW[1]) < cuav_util.gps_distance(nextWaypoint[0], nextWaypoint[1], nextnextW[0], nextnextW[1]):
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, -offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.crossBearing, self.deltaRowDist*2)
            else:
                self.SearchPattern.append(cuav_util.gps_newpos(nextnextW[0], nextnextW[1], self.searchBearing, offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                self.SearchPattern.append(cuav_util.gps_newpos(nextW[0], nextW[1], self.searchBearing, -offset))
                self.SearchPattern[-1] =(self.SearchPattern[-1][0], self.SearchPattern[-1][1], alt)
                #now turn 90degrees from bearing and width distance
                nextWaypoint = cuav_util.gps_newpos(nextW[0], nextW[1], self.crossBearing, self.deltaRowDist*2)

            #print "Next = " + str(nextWaypoint)

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
        print "Mean = " + str(meanPoint)

        if loiterInSearchArea == 1:
            tmp = self.entryPoints + self.SearchPattern + self.exitPoints
        else:
            tmp = self.entryPoints + self.SearchPattern + self.exitPoints

        return [(row[0], row[1]) for row in tmp]

    def getPolygonLength(self):
        '''Returns the search pattern path length (metres)'''
        distance = 0
        totPoint = self.entryPoints + self.SearchPattern + self.exitPoints

        for point in totPoint:
            if point != totPoint[-1]:
                distance = distance + cuav_util.gps_distance(point[0], point[1], totPoint[totPoint.index(point)-1][0], totPoint[totPoint.index(point)-1][1])

        return int(distance)

    def altitudeCompensation(self, heightAGL, numMaxPoints=100, threshold=5):
        '''Creates height points (ASL) for each point in searchArea,
        entry and exit points such that the plane stays a constant altitude above the ground,
        constrained by a max number of waypoints'''
        maxDeltaAlt = 0
        maxDeltaAltPoints = []
        maxDeltapercentAlong = 0

        EleModel = mp_elevation.ElevationModel()
        #make sure the SRTM tiles are downloaded
        EleModel.GetElevation(self.SearchPattern[0][0], self.SearchPattern[0][1])

        #get the ASL height of the airfield home, entry and exit points and initial search pattern
        # and add the heightAGL to them
        self.airportHeight = EleModel.GetElevation(self.airfieldHome[0], self.airfieldHome[1])
        self.airfieldHome = (self.airfieldHome[0], self.airfieldHome[1], heightAGL+EleModel.GetElevation(self.airfieldHome[0], self.airfieldHome[1]))

        for point in self.entryPoints:
            self.entryPoints[self.entryPoints.index(point)] = (point[0], point[1], heightAGL+EleModel.GetElevation(point[0], point[1]))

        for point in self.exitPoints:
            self.exitPoints[self.exitPoints.index(point)] = (point[0], point[1], heightAGL+EleModel.GetElevation(point[0], point[1]))

        for point in self.SearchPattern:
            self.SearchPattern[self.SearchPattern.index(point)] = (point[0], point[1], heightAGL+EleModel.GetElevation(point[0], point[1]))

        #keep looping through the search area waypoints and add new waypoints where needed 
        #to maintain const height above terrain
        print "---Starting terrain tracking optimisation---"
        while True:
            maxDeltaAlt = 0
            maxDeltaAltPoints = []
            maxDeltaPointIndex = 0

            for point in self.SearchPattern:
                if point != self.SearchPattern[-1]:
                    dist = cuav_util.gps_distance(point[0], point[1], self.SearchPattern[self.SearchPattern.index(point)+1][0], self.SearchPattern[self.SearchPattern.index(point)+1][1])
                    theta = cuav_util.gps_bearing(point[0], point[1], self.SearchPattern[self.SearchPattern.index(point)+1][0], self.SearchPattern[self.SearchPattern.index(point)+1][1])
                    AltDiff = float(self.SearchPattern[self.SearchPattern.index(point)+1][2] - point[2])
    
                    if numpy.around(theta) == numpy.around(self.searchBearing) or numpy.around(theta) == numpy.around((self.searchBearing+180) % 360):           
                        #increment 10% along waypoint-to-waypoint and get max height difference
                        for i in range(1, 9):
                            partPoint = cuav_util.gps_newpos(point[0], point[1], theta, (i*dist/10))
                            partAlt = EleModel.GetElevation(partPoint[0], partPoint[1]) + heightAGL
                            #print "Part = " + str(partAlt) + ", Orig = " + str(point[2])
                            if numpy.abs(point[2] + ((AltDiff*i)/10) - partAlt) > maxDeltaAlt:
                                maxDeltaAlt = numpy.abs(point[2] + ((AltDiff*i)/10) - partAlt)
                                maxDeltaAltPoint = (partPoint[0], partPoint[1], partAlt)
                                maxDeltaPointIndex = self.SearchPattern.index(point)+1
                                #print "New max alt diff: " + str(maxDeltaAlt) + ", " + str(partAlt)
            #now put a extra waypoint in between the two maxDeltaAltPoints
            self.SearchPattern.insert(maxDeltaPointIndex, maxDeltaAltPoint)
            #print "There are " + str(len(self.SearchPattern)) + " points in the search pattern. Max Alt error is " + str(maxDeltaAlt)
            if len(self.SearchPattern) >= numMaxPoints or threshold > maxDeltaAlt:
                break
        print "---Done terrain tracking optimisation---"


    def exportToMAVProxy(self, MAVpointLoader=None, loiterInSearchArea=1):
        '''Exports the airfield home, entry points, search pattern and exit points to MAVProxy'''

        #make a fake waypoint loader for testing purposes, if we're not
        #running within MAVProxy
        if MAVpointLoader is None:
            print "No loader - creating one"
            MAVpointLoader = mavwp.MAVWPLoader()

        entryjump = []
        exitjump = []

        #clear out all the old waypoints
        MAVpointLoader.clear()

        TargetSys = MAVpointLoader.target_system
        TargetComp = MAVpointLoader.target_component

        #Check the MAVLink version and handle appropriately
        if mavutil.mavlink10():
            fn = mavutil.mavlink.MAVLink_mission_item_message
        else:
            fn = mavutil.mavlink.MAVLink_waypoint_message

        # a dummy loiter waypoint
        dummyw = fn(TargetSys, TargetComp, 0,
                    MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, 0, 0, 0)

        #WP0 - add "airfield home" as waypoint 0. This gets replaced when GPS gets lock
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL, MAV_CMD_NAV_WAYPOINT, 1, 1, 0, 0, 0, 0, self.airfieldHome[0], self.airfieldHome[1], int(self.airfieldHome[2]))
        MAVpointLoader.add(w, comment='Airfield home')
        # form is fn(target_system=0, target_component=0, seq, frame=0/3, command=16, current=1/0, autocontinue=1, param1=0, param2=0, param3=0, param4=0, x, y, z)

        #WP1 - add in a jmp to entry lanes
        entryjump.append(MAVpointLoader.count())
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, 0, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to entry lane')
        MAVpointLoader.add(dummyw, 'jump dummy')

        #WP2 - takeoff, then jump to entry lanes
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_TAKEOFF, 0, 1, 10, 0, 0, 0, -26.585745, 151.840867, 30)
        MAVpointLoader.add(w, comment="Takeoff")
        entryjump.append(MAVpointLoader.count())
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, 0, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to entry lane')
        MAVpointLoader.add(dummyw, 'jump dummy')

        # landing approach
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, -26.592155, 151.842225, 80)
        MAVpointLoader.add(w, comment='Landing approach')

        # drop our speed
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_CHANGE_SPEED, 0, 1, 0, 22, 25, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Change to 22 m/s')

        # landing approach
        landing_approach_wpnum = MAVpointLoader.count()
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, -26.588218, 151.841345, 30)
        MAVpointLoader.add(w, comment='Landing approach 2')

        # drop our speed again
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_CHANGE_SPEED, 0, 1, 0, 18, 15, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Change to 18 m/s')

        # landing
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_LAND, 0, 1, 0, 0, 0, 0, -26.582821, 151.840247, 0)
        MAVpointLoader.add(w, comment='Landing')
        MAVpointLoader.add(dummyw, 'landing dummy')

        # comms Failure. Loiter at EL-1 for 2 minutes then fly to airfield home and loiter
        point = self.entryPoints[0]
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, point[0], point[1], int(point[2]-self.airportHeight))
        MAVpointLoader.add(w, comment='Comms Failure')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_LOITER_TIME, 0, 1, 120, 0, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='loiter 2 minutes')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, self.airfieldHome[0], self.airfieldHome[1], 90)
        MAVpointLoader.add(w, comment='Airfield home')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_LOITER_UNLIM, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='loiter')


        # GPS failure. Loiter in place for 30s then direct to airfield home
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_LOITER_TIME, 0, 1, 30, 0, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='GPS fail - loiter 30 secs')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, self.airfieldHome[0], self.airfieldHome[1], 90)
        MAVpointLoader.add(w, comment='Airfield home')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_LOITER_UNLIM, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='loiter')


        # joe drop approach
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, self.joeApproach[0], self.joeApproach[1], int(self.joeApproach[2]))
        MAVpointLoader.add(w, comment='Joe approach')
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, self.joeDrop[0], self.joeDrop[1], int(self.joeDrop[2]))
        MAVpointLoader.add(w, comment='Joe Drop location')

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_SET_SERVO, 0, 1, 7, 1776, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Drop bottle')

        # after drop, jump to exit lane
        exitjump.append(MAVpointLoader.count())        
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, 0, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to exit lane')
        MAVpointLoader.add(dummyw, 'jump dummy')

        #print "Done AF home"
        #WP12 - WPn - and add in the rest of the waypoints - Entry lane, search area, exit lane
        entry_wpnum = MAVpointLoader.count()
        for i in range(len(self.entryPoints)):
            point = self.entryPoints[i]
            w = fn(TargetSys, TargetComp, 0,
                   MAV_FRAME_GLOBAL_RELATIVE_ALT,
                   MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, point[0], point[1], int(point[2]-self.airportHeight))
            MAVpointLoader.add(w, comment='Entry %u' % (i+1))
        endentry_wpnum = MAVpointLoader.count()
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, 0, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to search mission')
        MAVpointLoader.add(dummyw, 'jump dummy')

        # exit points
        exit_wpnum = MAVpointLoader.count()
        for i in range(len(self.exitPoints)):
            point = self.exitPoints[i]
            w = fn(TargetSys, TargetComp, 0,
                   MAV_FRAME_GLOBAL_RELATIVE_ALT,
                   MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, point[0], point[1], int(point[2]-self.airportHeight))
            MAVpointLoader.add(w, comment='Exit point %u' % (i+1))

        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, landing_approach_wpnum, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to landing approach')
        MAVpointLoader.add(dummyw, 'jump dummy')

        # search pattern
        MAVpointLoader.wp(endentry_wpnum).param1 = MAVpointLoader.count()
        for i in range(len(self.SearchPattern)):
            point = self.SearchPattern[i]
            w = fn(TargetSys, TargetComp, 0,
                   MAV_FRAME_GLOBAL_RELATIVE_ALT,
                   MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0, point[0], point[1], int(point[2]-self.airportHeight))
            MAVpointLoader.add(w, comment='Search %u' % (i+1))

        #if desired, loiter in the search area for a bit
        if loiterInSearchArea == 1:
            meanPoint = tuple(numpy.average(self.SearchPattern, axis=0))
            w = fn(TargetSys, TargetComp, 0,
                   MAV_FRAME_GLOBAL_RELATIVE_ALT,
                   MAV_CMD_NAV_LOITER_TIME, 0, 1, 600, 0, 0, 0, meanPoint[0], meanPoint[1], int(meanPoint[2]-self.airportHeight))
            MAVpointLoader.add(w, comment='Loiter in search area for 10 minutes')

        exitjump.append(MAVpointLoader.count())        
        w = fn(TargetSys, TargetComp, 0,
               MAV_FRAME_GLOBAL_RELATIVE_ALT,
               MAV_CMD_DO_JUMP, 0, 1, 0, -1, 0, 0, 0, 0, 0)
        MAVpointLoader.add(w, comment='Jump to exit lane')
        MAVpointLoader.add(dummyw, 'jump dummy')

        # fixup jump waypoint numbers
        for wnum in entryjump:
            MAVpointLoader.wp(wnum).param1 = entry_wpnum
        for wnum in exitjump:
            MAVpointLoader.wp(wnum).param1 = exit_wpnum

        #export the waypoints to a MAVLink compatible format/file
        waytxt = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', "way.txt")
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

    def getCameraWidth(self, alt):
        '''Using the camera parameters, with the width of the
        ground strip that the camera can see from a particular altitude'''

        #use the camera parameters
        c_params = CameraParams(lens=4.0)
        # load camera params and get the width of an image on the ground
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'chameleon1_arecont0.json')
        c_params.load(path)
        aov = math.degrees(2.0*math.atan((c_params.sensorwidth/1000.0)/(2.0*c_params.lens/1000.0)))
        groundWidth = 2.0*alt*math.tan(math.radians(aov/2))

        return groundWidth


if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser("mp_missiongenerator.py [options]")
    parser.add_option("--file", type='string', default='..//data//OBC Waypoints.kml', help="input file")
    parser.add_option("--searchAreaMask", type='string', default='SA-', help="name mask of search area waypoints")
    parser.add_option("--missionBoundaryMask", type='string', default='MB-', help="name mask of mission boundary waypoints")
    parser.add_option("--searchAreaOffset", type='int', default=10, help="distance waypoints will be placed outside search area")
    parser.add_option("--wobble", type='int', default=1, help="Make every second row slightly offset. Aids in viewing the overlaps")
    parser.add_option("--width", type='int', default=0, help="Width (m) of each scan row. 0 to use camera params")
    parser.add_option("--overlap", type='int', default=50, help="% overlap between rows")
    parser.add_option("--entryLane", type='string', default='EL-1,EL-2', help="csv list of waypoints before search")
    parser.add_option("--exitLane", type='string', default='EL-3,EL-4', help="csv list of waypoints after search")
    parser.add_option("--altitude", type='int', default=90, help="Altitude of waypoints")
    parser.add_option("--terrainTrack", type='int', default=1, help="0 if --altitude is ASL, 1 if AGL (terrain tracking)")
    parser.add_option("--loiterInSearchArea", type='int', default=1, help="1 if UAV loiters in search area at end of search. 0 if it goes home")

    (opts, args) = parser.parse_args()

    gen = MissionGenerator(opts.file)
    gen.Process(opts.searchAreaMask, opts.missionBoundaryMask)
    gen.CreateEntryExitPoints(opts.entryLane, opts.exitLane)

    groundWidth = opts.width
    #are we using the camera params to get the size of each search strip?
    if opts.width == 0:
        groundWidth = gen.getCameraWidth(opts.altitude)
    print "Strip width = " + str(groundWidth)

    gen.CreateSearchPattern(width = groundWidth, overlap=opts.overlap, offset=opts.searchAreaOffset, wobble=opts.wobble, alt=opts.altitude)
    
    #are we using terrain tracking?
    if opts.terrainTrack == 1:
        gen.altitudeCompensation(heightAGL = opts.altitude)
    gen.ExportSearchPattern()

    #start a map
    sm = mp_slipmap.MPSlipMap(lat=gen.getMapPolygon(loiterInSearchArea=opts.loiterInSearchArea)[0][0], lon=gen.getMapPolygon(loiterInSearchArea=opts.loiterInSearchArea)[0][1], elevation=True, service='GoogleSat')
    sm.add_object(mp_slipmap.SlipPolygon('Search Pattern', gen.getMapPolygon(), layer=1, linewidth=2, colour=(0,255,0)))

    #get the search pattern distance
    print "Total Distance = " + str(gen.getPolygonLength()) + "m"

    #and export to MAVProxy
    gen.exportToMAVProxy(MAVpointLoader=None, loiterInSearchArea=opts.loiterInSearchArea)

    #and to google earth
    gen.ExportSearchPattern()





