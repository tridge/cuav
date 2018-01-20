#!/usr/bin/env python3

'''
Download and sync image data from the cuav S3 repository
This will be a lot of data. Be careful!
'''

import os
import csv
import subprocess
import argparse

#The S3 bucket we're downloading from
cuavBucket = "canberrauav2017imagerystorage"


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUAV image database sync')
    parser.add_argument('path', help= 'local data storage folder')
    args = parser.parse_args()
    
    print("CanberraUAV image data upload")
    
    #check the output folder
    if not os.path.exists(args.path):
        print("Not a valid local directory:")
        print(os.path.abspath(args.path))
        exit(0)

    #grab the credentials
    (username, accesskey, secretkey) = loadCredentials('credentials.csv')

    print("Using user: " + username)

    #Just use a subprocess for this (with the AWS env vars set)
    my_env = os.environ.copy()
    my_env["AWS_ACCESS_KEY_ID"] = accesskey
    my_env["AWS_SECRET_ACCESS_KEY"] = secretkey
    sub = subprocess.Popen("aws s3 sync s3://"+cuavBucket +" /home/stephen/Documents/tmmp", env=my_env, shell=True)
    sub.communicate()
    print("Finished!")
    
    
    
