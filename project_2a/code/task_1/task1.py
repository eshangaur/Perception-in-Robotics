#!usr/bin/python

# This file uses python 3.7.4 and OpenCV

import cv2
import glob

if __name__ == '__main__':      # if we are running this file directly
    # STEP 1: read in all images to memory
    left_ = [cv2.imread(image) for image in glob.glob("../../images/task_1/left_*.png")]
    right = [cv2.imread(image) for image in glob.glob("../../images/task_1/right_*.png")]

    # STEP 2: Extract 3D to 2D point correspondences
    
    