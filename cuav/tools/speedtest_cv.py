#!/usr/bin/env python

import sys, cv2, time, os, glob
import argparse
import numpy as np

def do_speedtest(filename):
    '''show edges in an image'''
    infile = cv2.imread(filename,-1)

    v = np.median(infile)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 - sigma) * v))
    
    cv2.Canny(infile, lower, upper)


def circle_highest(filename):
    '''circle the highest value pixel in an image'''
    infile = cv2.imread(filename,-1)
    
    gray_image = cv2.cvtColor(infile, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
    cv2.circle(infile, max_loc, 10, (0,0,255), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCV Speedtest for performance comparison")
    parser.add_argument("files", default=None, help="Image directory or example images")
    args = parser.parse_args()
    
    files = []
    types = ('*.png', '*.jpeg', '*.jpg')
    if os.path.isdir(args.files):
        for tp in types:
            files.extend(glob.glob(os.path.join(args.files, tp)))
    else:
        files.append(args.files)

    t1 = time.time()
    for f in files:
        image = do_speedtest(f)
    t2 = time.time()
    print("Edges: Processed %u images in %f seconds - %f fps" % (len(files), t2-t1, len(files)/(t2-t1)))


    t1 = time.time()
    for f in files:
        image = circle_highest(f)
    t2 = time.time()
    print("Highest: Processed %u images in %f seconds - %f fps" % (len(files), t2-t1, len(files)/(t2-t1)))

