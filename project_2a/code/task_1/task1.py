#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


import cv2
import glob
import numpy as np
from IPython.display import Image


# ## Task 1: Pinhole camera model and calibration
# https://docs.google.com/document/d/1ICOuXPzNbGSG0eNyMCZUArUQ9a8uhqBODID1ZDELYFw/edit#heading=h.rahbroakbdke

# ### Step (1): Load the images. 
# Please use the images in the provided resource files. For this task, the folder is "images/task_1". Since a stereo camera system is used, there are two sets of images with prefixes of "left_" and "right_", indicating which camera took the images. You are going to calibrate each individual camera separately, i.e., if you want to calibrate the left camera, use those images with prefixes of "left_". You can use the OpenCV library function "imread()" for this step.
# 

# In[3]:


# left_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_1/left_*.png"))]
# right_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_1/right_*.png"))]

left_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/opencv_sample_calibration_images/left*"))]
right_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/opencv_sample_calibration_images/right*"))]

import matplotlib.pyplot as plt
def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    if(nrows > 1 or ncols > 1):
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optional
    else:
        for ind,title in enumerate(figures):
            axeslist.imshow(figures[title], cmap=plt.gray())
            axeslist.set_title(title)
            axeslist.set_axis_off()
            

plot_figures({'left_1': left_[0], 'right_1': right_[0]}, 1, 2)


# In[4]:


left_[0].shape


# ### Step (2): Extract 3D-to-2D point correspondences. 
# For each image with the calibration board pattern, you are going to extract the corner points on the image using OpenCV library function `findChessboardCorners()`. These are the 2D points. The 3D points are just (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (n, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), ..., (n, 1, 0), (0, 2, 0), ..., (n, 2, 0), ..., (n, m, 0), where the number of calibration board corners are n by m. These 3D points use a reference frame on the board with one grid cell as unit length. Note that the provided images use a calibration board of 9 by 6, and each square cell has a width and height of 25.4 mm, i.e., exactly one inch. If you want to use the actual scale, you can just multiply the 3D coordinates with the actual cell length (e.g., in meters or millimeters). However, a single camera can not reliably recover the scale of the actual imaged objects. Also note that the detected corner points on the board are sorted row by row in a numpy array in Python or an STL vector in C++ (from red to blue if you draw them on the image), so the correspondences are already there as the 3D points are known. Note that you may use all the images for calibration or just a subset of them. Usually 3 to 10 images are enough, but using only one image may not provide satisfying results.
# 

# ### Example

# In[5]:


retval, corners_2D = cv2.findChessboardCorners(left_[0],(6,9))
# print(retval,corners_2D)

corner_img = left_[0].copy()
cv2.drawChessboardCorners(corner_img,(6,9),corners_2D,retval)
plot_figures({'left_0 with corners': corner_img})


# ## Calibreate left and right cameras individually

# In[6]:


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

rows = 6
columns = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(columns,5,0)
objp = np.zeros((columns*rows,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
# objp = objp*25.4   # only use when using the images from Duo

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
breakPoint = 69


# In[7]:


imgpoints_left = [] # 2d points in image plane.

for i, img_og in enumerate(left_):
    img = img_og.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns,rows),None)
    print(ret,end=' ')
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)   # this does not do anything
        imgpoints_left.append(corners2)
    
    if i==breakPoint:
        break


# In[8]:


imgpoints_right = [] # 2d points in image plane.

for i, img_og in enumerate(right_):
    img = img_og.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns,rows),None)
    print(ret,end=' ')
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)   # this does not do anything
        imgpoints_right.append(corners2)
        
    if i==breakPoint:
        break


# ### Step (3): Calculate camera intrinsic parameters.
# Once the 3D-to-2D point correspondences are obtained, call OpenCV library function [`calibrateCamera()`](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera) to calculate the camera intrinsic matrix and distort coefficients. 

# In[9]:


ret, mtx_left, distCoeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, (640, 480),None,None)

ret, mtx_right, distCoeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, (640, 480),None,None)


# In[10]:


len(rvecs_left), len(tvecs_left)


# In[11]:


rvecs_left


# In[12]:


mtx_left


# ### Step (4): Check the calibration results.
# After the camera parameters are obtained, you can undistort the images of calibration board patterns with these parameters using OpenCV library function "initUndistortRectifyMap()" and "remap()". An example is shown in Figure 3. Note that on the undistorted image, the lines of the chessboard pattern are straight. This is expected since 3D-to-2D projective transformation maintains line straightness.

# In[23]:


img = left_[0].copy()
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,distCoeffs_left,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx_left, distCoeffs_left, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst_crop = dst[y:y+h, x:x+w]

print("\t\t\t\t left_0.png results")
plot_figures({'og':img, 'undistorted':dst, 'undst_crop':dst_crop},1,3)


# In[24]:


img = left_[2].copy()
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,distCoeffs_left,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx_left, distCoeffs_left, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst_crop = dst[y:y+h, x:x+w]

print("\t\t\t\t left_2.png results")
plot_figures({'og':img, 'undistorted':dst, 'undst_crop':dst_crop},1,3)


# In[21]:


img = right_[2].copy()
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,distCoeffs_left,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx_left, distCoeffs_left, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst_crop = dst[y:y+h, x:x+w]

print("\t\t\t\t right_2.png results")
plot_figures({'og':img, 'undistorted':dst, 'undst_crop':dst_crop},1,3)


# ### Step (5): Save the camera intrinsic parameters to a file.
# You can use OpenCV "FileStorage" class to write the intrinsic matrix and distort coefficients into a single XML file, or you can just use NumPy "savetxt()" to write them into multiple CSV files if you choose to use Python. These parameters should be saved for future use. In the next tasks, You can use the same OpenCV "FileStorage" class to read out the saved camera parameters.

# In[16]:


s = cv2.FileStorage('../../parameters/left_camera_intrinsics.xml', cv2.FileStorage_WRITE)

s.write('mtx_left', mtx_left)
s.write('distCoeffs_left', distCoeffs_left)
# s.write('rvecs_left', rvecs_left)
# s.write('tvecs_left', tvecs_left)

s.release()


s = cv2.FileStorage('../../parameters/right_camera_intrinsics.xml', cv2.FileStorage_WRITE)

s.write('mtx_right', mtx_right)
s.write('distCoeffs_right', distCoeffs_right)
# s.write('rvecs_right', rvecs_right)
# s.write('tvecs_right', tvecs_right)

s.release()


# In[40]:


sorted(glob.glob("../../images/opencv_sample_calibration_images/left*"))[1][46:]


# In[30]:


import os


# In[49]:


paths = sorted(glob.glob("../../images/opencv_sample_calibration_images/left*"))
output_path = '../../output/task_1/'

for i, img_og in enumerate(left_):
    img = img_og.copy()
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,distCoeffs_left,(w,h),1,(w,h))

    dst = cv2.undistort(img, mtx_left, distCoeffs_left, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst_crop = dst[y:y+h, x:x+w]

    path = paths[i][18:]
    cv2.imwrite(os.path.join(output_path, 'undistorted_' + paths[i][46:]), dst_crop)


# In[51]:


paths = sorted(glob.glob("../../images/opencv_sample_calibration_images/right*"))
output_path = '../../output/task_1/'

for i, img_og in enumerate(right_):
    img = img_og.copy()
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_left,distCoeffs_left,(w,h),1,(w,h))

    dst = cv2.undistort(img, mtx_left, distCoeffs_left, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst_crop = dst[y:y+h, x:x+w]

    path = paths[i][18:]
    cv2.imwrite(os.path.join(output_path, 'undistorted_' + paths[i][46:]), dst_crop)


# In[ ]:




