#!/usr/bin/python
'''
Outback Challenge Mission Generator
It read in a KML file and generate mission waypoints (ie. search pattern)
Created by Stephen Dade (stephen_dade@hotmail.com)
'''

import numpy, os, time, sys, xml.dom.minidom

class MissionGenerator():
    '''Mission Generator Class'''

    def __init__(self, inFile='obc.kml'):
        self.inFile = inFile
        self.dom = xml.dom.minidom.parse(inFile)

    def Process(self):
        '''Processes the imported xml file for points'''
        
        self.searchArea = []
        self.missionBounds = []
        #get a list of all points in the kml file:
        airf = self.dom.getElementsByTagName('Placemark')
        for point in airf:
            if self.getElement(point.getElementsByTagName('name')[0]) == "Airfield Home":
                self.airfieldHome = (float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0])))
                print "Airfield Home = " + str(self.airfieldHome)
            if "SA-" in self.getElement(point.getElementsByTagName('name')[0]):
                self.searchArea.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))
            if "MB-" in self.getElement(point.getElementsByTagName('name')[0]):
                self.missionBounds.append((float(self.getElement(point.getElementsByTagName('latitude')[0])), float(self.getElement(point.getElementsByTagName('longitude')[0]))))

        print "Search Area = " + str(self.searchArea)
        print "Mission Boundary = " + str(self.missionBounds)

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
        #print "Bounding box is: " + str(self.boundingBoxLat) + " ... " + str(self.boundingBoxLong)

        #for point in self.searchArea:
        #    print self.getDistAndBearing(self.airfieldHome, point)


    def CreateSearchPattern(self, width=50.0, overlap=10.0):
        '''Generate the waypoints for the search pattern, using alternating strips
        width is the width (m) of each strip
        overlap is the % overlap between strips'''
        self.SearchPattern = []

        #find the nearest point to Airfield Home - use this as a starting point
        nearestdist = self.getDistAndBearing(self.airfieldHome, self.searchArea[0])[0]
        nearest = self.searchArea[0]
        for point in self.searchArea:
            newdist = self.getDistAndBearing(self.airfieldHome, point)[0]
            if newdist < nearestdist:
                nearest = point
                nearestdist = newdist

        print "Start = " + str(nearest) + ", dist = " + str(nearestdist)

        #the search pattern will then run parallel between the two furthest points in the list
        searchLine = (0, 0)
        for point in self.searchArea: 
            newdist = self.getDistAndBearing(point, self.searchArea[self.searchArea.index(point)-1])
            if newdist[0] > searchLine[0]:
                searchLine = newdist

        self.searchBearing = searchLine[1]
        print "Search bearing is " + str(self.searchBearing) + "/" + str((self.searchBearing + 180) % 360)

        #need to find the 90 degree bearing to searchBearing that is inside the search area. This
        #will be the bearing we increment the search rows by
        if not self.polygon_outside(self.addDistBearing(nearest, 10, (self.searchBearing + 45) % 360), self.searchArea):
            self.crossBearing = (self.searchBearing + 90) % 360
        else:
            self.crossBearing = (self.searchBearing - 90) % 360
        print "Cross bearing is: " + str(self.crossBearing)

        #function check
        newposn = self.addDistBearing(nearest, nearestdist, 358)
        print "Going back to start: " + str(newposn)

        #the distance between runs is this:
        self.deltaRowDist = width - width*(overlap/100)
        if self.deltaRowDist <= 0:
            print "Error, overlap % is too high"
            return

        #we are starting at the "nearest" and mowing the lawn parallel to "self.searchBearing"
        #first waypoint is width*(overlap/100) on an opposite bearing (so behind the search area)
        nextWaypoint =  self.addDistBearing(nearest, width*(overlap/100), self.crossBearing)
        print "First = " + str(nextWaypoint)
        #self.SearchPattern.append(firstWaypoint)

        #mow the lawn, every 2nd row:
        while True:
            (nextW, nextnextW) = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            if self.getDistAndBearing(nextWaypoint, nextW)[0] < self.getDistAndBearing(nextWaypoint, nextnextW)[0]:
                self.SearchPattern.append(nextW)
                self.SearchPattern.append(nextnextW)
            else:
                self.SearchPattern.append(nextnextW)
                self.SearchPattern.append(nextW)

            #now turn 90degrees from bearing and width distance, from the midpoint of the prev 2 points
            nextWaypoint = ((nextW[0]+nextnextW[0])/2, (nextW[1]+nextnextW[1])/2) 
            nextWaypoint = self.addDistBearing(nextWaypoint, width*2, self.crossBearing)
            print "Next = " + str(nextWaypoint)
            #self.SearchPattern.append(nextWaypoint)
            if self.isOutsideSearchAreaBoundingBox(nextWaypoint[0], nextWaypoint[1]):
                print "Stopped = " + str(nextWaypoint)
                break
        
        #go back and do the rows we missed
        self.crossBearing = (self.crossBearing + 180) % 360
        nextWaypoint = self.addDistBearing(self.SearchPattern[-1], width, self.crossBearing)
        while True:
            (nextW, nextnextW) = self.projectBearing(self.searchBearing, nextWaypoint, self.searchArea)
            if self.getDistAndBearing(nextWaypoint, nextW)[0] < self.getDistAndBearing(nextWaypoint, nextnextW)[0]:
                self.SearchPattern.append(nextW)
                self.SearchPattern.append(nextnextW)
            else:
                self.SearchPattern.append(nextnextW)
                self.SearchPattern.append(nextW)

            #now turn 90degrees from bearing and width distance
            nextWaypoint = ((nextW[0]+nextnextW[0])/2, (nextW[1]+nextnextW[1])/2) 
            nextWaypoint = self.addDistBearing(nextWaypoint, width*2, self.crossBearing)
            print "Next = " + str(nextWaypoint)
            #self.SearchPattern.append(nextWaypoint)
            if self.isOutsideSearchAreaBoundingBox(nextWaypoint[0], nextWaypoint[1]):
                break
        print("here")

    def isOutsideSearchAreaBoundingBox(self, lat, longy):
        '''Checks if the long/lat pair is inside the search area
        bounding box. Returns true if it is inside'''
        if lat < self.boundingBoxLat[0]:
            return 1
        if lat > self.boundingBoxLat[1]:
            return 1
        if longy < self.boundingBoxLong[0]:
            return 1
        if longy > self.boundingBoxLong[1]:
            return 1
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
                f.write("            <coordinates>" + str(point[1]) + "," + str(point[0]) + ",0</coordinates>\n")
                f.write("        </Point>\n")
                f.write("    </Placemark>\n")

            #make a polygon for easier viewing
            f.write("    <Placemark>\n")
            f.write("        <name>Search Pattern</name>\n")
            f.write("        <LineString>\n")
            f.write("            <tessellate>0</tessellate>\n")
            f.write("            <coordinates>\n")
            for point in self.SearchPattern:
                f.write("                " + str(point[1]) + "," + str(point[0]) + ",0\n")
            f.write("            </coordinates>\n")
            f.write("        </LineString>\n")
            f.write("    </Placemark>\n")
            
            f.write("</Document>\n")
            f.write("</kml>")


    def getElement(self, element):
        return self.getText(element.childNodes)


    def getText(self, nodelist):
        rc = ""
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc = rc + node.nodeValue
        return rc


    def getDistAndBearing(self, startPosn, endPosn):
        '''Returns the distance (m) and bearing (deg from north) from a lat/long tuple to
        an ending lat/long tuple'''

        #use the sperical law of cosines formula for distance
        dist = numpy.arccos(numpy.sin(numpy.radians(startPosn[0]))*numpy.sin(numpy.radians(endPosn[0]))+ \
               numpy.cos(numpy.radians(startPosn[0]))*numpy.cos(numpy.radians(endPosn[0]))* \
               numpy.cos(numpy.radians(endPosn[1])-numpy.radians(startPosn[1])))*6372.79*1000

        #this formula taken from www.movable-type.co.uk/scripts/latlong.html
        bearing = numpy.arctan2(numpy.sin(numpy.radians(endPosn[1])-numpy.radians(startPosn[1]))*numpy.cos(numpy.radians(endPosn[0])), \
                                numpy.cos(numpy.radians(startPosn[0]))*numpy.sin(numpy.radians(endPosn[0])) - \
                                numpy.sin(numpy.radians(startPosn[0]))*numpy.cos(numpy.radians(endPosn[0]))* \
                                numpy.cos(numpy.radians(endPosn[1])-numpy.radians(startPosn[1])))

        #format bearing to a degrees compass heading
        bearing = (numpy.degrees(bearing) + 360) % 360

        return (dist, bearing)

    def addDistBearing(self, startPosn, dist, bearing):
        '''Travels dist and bearing from the startPosn (lat/long tuple) and
        returns the ending posn (lat/long tuple)'''

        #this formula taken from www.movable-type.co.uk/scripts/latlong.html
        lat = numpy.arcsin(numpy.sin(numpy.radians(startPosn[0]))*numpy.cos(dist/(6372.79*1000))+numpy.cos(numpy.radians(startPosn[0]))*numpy.sin(dist/(6372.79*1000))*numpy.cos(numpy.radians(bearing)))
        deltaLong = numpy.arctan2(numpy.sin(numpy.radians(bearing))*numpy.sin(dist/(6372.79*1000))*numpy.cos(numpy.radians(startPosn[0])), \
                    numpy.cos(dist/(6372.79*1000))-numpy.sin(numpy.radians(startPosn[0]))*numpy.sin(lat))
        longitude = numpy.radians(startPosn[1]) + deltaLong

        return (numpy.degrees(lat), numpy.degrees(longitude))

    def polygon_outside(self, P, V):
	    '''return true if point is outside polygon
	    P is a (x,y) tuple
	    V is a list of (x,y) tuples
        
	    The point in polygon algorithm is based on:
	    http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
	    '''
	    n = len(V)
	    outside = True
	    j = n-1
	    for i in range(n):
		    if (((V[i][1]>P[1]) != (V[j][1]>P[1])) and
		        (P[0] < (V[j][0]-V[i][0]) * (P[1]-V[i][1]) / (V[j][1]-V[i][1]) + V[i][0])):
			    outside = not outside
		    j = i
	    return outside

    def projectBearing(self, bearing, startPos, searchArea):
        '''Projects bearing from startPos until it reaches the edge(s) of
        searchArea (list of lat/long tuples. Returns the First/Last position(s)'''

        #the resolution of the search pattern (m)
        delta = 5

        #first, project along until we reach the edge of the bounding box
        BoundaryWaypoint = self.addDistBearing(startPos, delta, bearing)
        while self.isOutsideSearchAreaBoundingBox(BoundaryWaypoint[0], BoundaryWaypoint[1]) == 0:
             BoundaryWaypoint = self.addDistBearing(BoundaryWaypoint, delta, bearing)

        #now project back until we're just outside the search area
        FirstWaypoint = BoundaryWaypoint
        while self.polygon_outside(FirstWaypoint, self.searchArea):
             FirstWaypoint = self.addDistBearing(FirstWaypoint, delta, (bearing + 180) % 360)
        FirstWaypoint = self.addDistBearing(FirstWaypoint, delta, bearing)

        #and keep going until we're at the other end of the search area
        LastWaypoint = self.addDistBearing(FirstWaypoint, delta, (bearing + 180) % 360)
        while self.polygon_outside(LastWaypoint, self.searchArea) == 0:
             LastWaypoint = self.addDistBearing(LastWaypoint, delta, (bearing + 180) % 360)

        #and return
        return (FirstWaypoint, LastWaypoint)


if __name__ == "__main__":

    from optparse import OptionParser
    parser = OptionParser("mp_missiongenerator.py [options]")
    parser.add_option("--file", type='string', default='obc.kml', help="input file")

    (opts, args) = parser.parse_args()

    inFile = opts.file

    gen = MissionGenerator(inFile)
    gen.Process()
    gen.CreateSearchPattern()
    gen.ExportSearchPattern()




