#!/usr/bin/env python3

'''
Upload the latest image search data to AWS S3
'''

import os
import boto3
import csv
import time

#Where the image data is being sourced from
imageFolder = os.path.expanduser('~/images_captured')

#The S3 bucket we're uploading to
cuavBucket = "canberrauav2017imagerystorage"

#Function to get last-modified folder
def all_subdirs_of(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

#Loads the AWS credentials
#Returns tuple of (username, access key, secret key)
#or exits if it failed
def loadCredentials(infile):
    try:
        with open(infile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                uname = row['User name']
                akey = row ['Access key ID']
                seckey = row ['Secret access key']
            return (uname, akey, seckey)
    except:
        print("Unable to load credentials.csv")
        exit(0)

#returns the latest modified subfolder
def getUploadDir(basedir):
    all_subdirs = all_subdirs_of(basedir)
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return os.path.join(imageFolder, latest_subdir)

#returns the size of a folder in MB
def getFoldersize(infolder):
    return sum( os.path.getsize(os.path.join(dirpath,filename)) for dirpath, dirnames, filenames in os.walk( infolder ) for filename in filenames ) / (1024*1024)



if __name__ == '__main__':
    print("CanberraUAV image data upload")

    #grab the credentials
    (username, accesskey, secretkey) = loadCredentials('credentials.csv')

    print("Using user: " + username)
    #Start the AWS S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=accesskey,
        aws_secret_access_key=secretkey,
    )
    
    #Get flight details
    #YYMMDD-Location-Pilot-Platform-Mission
    print("Enter the location of the flight (ie. CMAC)")
    missionlocation = input(">")
    print("Enter the name of the Pilot (ie. Stephen)")
    pilotname = input(">")
    print("Enter the name of the Platform (ie. Opterra)")
    airframename = input(">")
    print("Enter the Mission Number (ie. 1)")
    missionname = input(">")
    
    # Call S3 to list current buckets
    response = s3.list_buckets()

    # Get a list of all bucket names from the response
    buckets = [bucket['Name'] for bucket in response['Buckets']]

    # Look for cuav bucket
    if cuavBucket in buckets:
        print("Connected to AWS")
    else:
        print("Cannot connect to AWS")
        exit(0)

    #figure out which folder to upload
    folderToUpload = getUploadDir(imageFolder)
    print("Ready to upload folder " + folderToUpload + ", which is {0:.2f} MB".format(getFoldersize(folderToUpload)))
    
    #Create the S3 folder name
    s3name = time.strftime('%Y%m%d', time.localtime(os.path.getmtime(folderToUpload))) + "-" + missionlocation + "-" + pilotname + "-" + airframename + "-" + missionname
    print("S3 folder will be " + s3name)

    print("Ready to upload? (y/n)")
    if input(">") != "y":
        exit(0)

    #and actually upload (file-by-file)
    numfiles = len([name for name in os.listdir(folderToUpload) if os.path.isfile(os.path.join(folderToUpload, name))])
    curfile = 1
    for root, dirs, files in os.walk(folderToUpload):

      for filename in files:

        # construct the full local path
        local_path = os.path.join(root, filename)

        # construct the full S3 path
        relative_path = os.path.relpath(local_path, folderToUpload)
        s3_path = os.path.join(s3name, relative_path)

        print("Uploading \"%s\" (%s/%s)" % (s3_path, curfile, numfiles))
        s3.upload_file(local_path, cuavBucket, s3_path)
        curfile = curfile + 1

    print("Upload complete")
