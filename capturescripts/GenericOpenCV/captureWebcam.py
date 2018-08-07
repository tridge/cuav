#!/usr/bin/env python
'''
An example program to capture cuav-compliant images using OpenCV.
NOTE: The capture time of the images (represented in the filename) must be <100ms
precise, otherwise the geolocation will be inaccurate

This uses Python 2.7/3.x and OpenCV2

Use "sudo apt-get install opencv-python" to install OpenCV2
'''
import time, cv2, os

def frame_time(t):
    '''return a time string for a filename with 0.01 sec resolution'''
    # round to the nearest 100th of a second
    t += 0.005
    hundredths = int(t * 100.0) % 100
    return "%s%02uZ" % (time.strftime("%Y%m%d%H%M%S", time.gmtime(t)), hundredths)

if __name__ == '__main__':
    #Open the default camera on the system
    camera = cv2.VideoCapture(0)

    #Set the camera resolution
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)

    #grab an image just to activate the camera
    camera.grab()

    #And keep capturing, (until CTRL+C) with timestamp
    try:
        while True:
            #grab the image and get timestamp
            camera.grab()
            frametime = frame_time(time.time())
            
            #Get the grabbed image from the device and write to file as png
            return_value, image = camera.retrieve()
            cv2.imwrite(frametime + '.jpg', image)
            
            #Symlink to the latest-written image for cuav processing
            try:
                os.unlink('camera.jpg')
            except OSError:
                pass
            os.symlink(frametime + '.jpg', 'camera.jpg')
            print("Got: " + frametime + '.jpg')
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    #Finally, shutdown the camera
    print("Closing camera")
    del(camera)

