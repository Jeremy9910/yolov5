import sys
import argparse
import cv2


def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        if success == False:
            break
        cv2.imwrite( pathOut + "%03d.jpg" % count, image)    
        count = count + 5
    

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", default="VID_20221205_161724.mp4" ,help="path to video")
    a.add_argument("--pathOut", default="/home/kav/dev/VideoConverter-main/ordner10/",help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)