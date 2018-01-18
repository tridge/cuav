#!/usr/bin/env python3

'''
Enter Joe coordinates in text
'''

import os

#Python 3 fix

#File format is:
#lat, lon
#lat, lon

#Get the high-level output folder
outputFolder = os.path.expanduser('~/images_captured')

#Function to get last-modified folder
def all_subdirs_of(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

#keep asking user for Joe coordinates
while(True):
    #Figure out the output file
    all_subdirs = all_subdirs_of(outputFolder)
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    outfile = os.path.join(outputFolder, latest_subdir, "joe.txt")
    print("Output file is " + outfile)
    
    fLat = -4000
    fLon = -4000
    print("Enter Joe coordinates in lat,lon")
    coords = input('>')
    
    #try-parse
    [Slat, Slon] = coords.split(',')
    try:
        fLat = float(Slat)
        fLon = float(Slon)
    except ValueError:
        print("Invalid value, try again")
    if(fLat == -4000 or fLon == -4000):
        print("Invalid values")
    else:
        print("Values OK")
        with open(outfile, 'a') as f:
            f.write(str(fLat) + ", " + str(fLon) + "\n")
        
    #ask user if they want to enter more
    print("Enter another position? (y/n)")
    more = input('>')
    if more == "n":
        break


print("End program")
