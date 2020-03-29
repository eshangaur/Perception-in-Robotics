#!/usr/bin/env python
# coding: utf-8

# # Setup
# [Project Description](https://docs.google.com/document/d/1U7U4PN39LvbMh9_PGCAjtJU7BS_omKqtc081Qqbcvlg/edit)

# ## imports

# In[1]:


import cv2
import glob
import numpy as np
import sys
print(sys.version)
print("OpenCV version :  {0}".format(cv2.__version__))


# In[2]:


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


# ## Camera parameter matrices

# In[3]:


left_intrinsic = np.array([[423.27381306, 0, 341.34626532],
                           [0, 421.27401756, 269.28542111],
                           [0, 0, 1]])

right_intrinsic = np.array([[420.91160482, 0, 352.16135589],
                            [0, 418.72245958, 264.50726699],
                            [0, 0, 1]])

distCoeffs_left = np.array([-0.43394157423038077, 0.26707717557547866,
                             -0.00031144347020293427, 0.0005638938101488364,
                             -0.10970452266148858])
distCoeffs_right = np.array([-0.4145817681176909, 0.19961273246897668,
                             -0.00014832091141656534, -0.0013686760437966467,
                             -0.05113584625015141])


# # Step 1: Load the images

# In[4]:


left_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_6/left_*.png"))]
right_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_6/right_*.png"))]

len(left_), len(right_)


# In[5]:


# plot_figures({'left_0': left_[0], 'right_0': right_[0], 'left_1': left_[1], 'right_1': right_[1]}, 2, 2)


# # Step 2: Detect ArUco markers

# In[6]:


from cv2 import aruco

dic = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

markers = aruco.detectMarkers(left_[0],dic)   # detect the marker on a single image

temp = aruco.drawDetectedMarkers(left_[0].copy(), markers[0])
# plot_figures({'detected markers':temp}, 1,1)


# ## Create output for Duo

# In[8]:


for i, img in enumerate(left_):
    markers = aruco.detectMarkers(img,dic)   # detect the marker on a single image

    temp = aruco.drawDetectedMarkers(img.copy(), markers[0])
    cv2.imwrite('../../output/task_6/left_%d_aruco_marker.png' % i, temp)


# # Step 3: Estimate Camera Pose
# [solvePnP()](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#bool%20solvePnP(InputArray%20objectPoints,%20InputArray%20imagePoints,%20InputArray%20cameraMatrix,%20InputArray%20distCoeffs,%20OutputArray%20rvec,%20OutputArray%20tvec,%20bool%20useExtrinsicGuess,%20int%20flags))

# In[9]:


import math


# In[10]:


objPoints = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],dtype=np.float64)
objPoints.shape


# In[11]:


markers[0][0][0]


# In[12]:


markers[0][0][0].shape


# In[13]:


# rotation vector and translation vector of the camera relative to the objectPoints
retval, rot_vec, trans_vec = cv2.solvePnP(objPoints, markers[0][0][0], left_intrinsic, distCoeffs_left)
retval


# In[14]:


trans_vec*4


# # Step 4: Check camera pose

# ### Functions and other

# In[15]:


import math

def create3DPlot(withAxes=True, autoscale=False, bound=5):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    
    if not autoscale:
        ax.autoscale(False)
        ax.set_xbound(-bound,bound)
        ax.set_ybound(-bound,bound)
        ax.set_zbound(-bound,bound)
        
    if withAxes:
        # x axis is red
        x_line = np.linspace(0,4,100)
        y_line = x_line*0
        z_line = y_line
        ax.plot3D(x_line, y_line, z_line,'red')

        # y axis is green
        y_line = np.linspace(0,4,100)
        x_line = y_line*0
        z_line = x_line
        ax.plot3D(x_line, y_line, z_line,'green')

        # z axis is blue
        z_line = np.linspace(0,4,100)
        y_line = z_line*0
        x_line = y_line
        ax.plot3D(x_line, y_line, z_line,'blue')
        
    return ax


# In[16]:


# example rotation
# ax = create3DPlot()
# x_points = trans_vec[0]
# y_points = trans_vec[1]
# z_points = trans_vec[2]
# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');


# rot_point = trans_vec+(rot_vec*10)
# rot_point = Quaternion(axis=[0,0,1],angle=math.radians(-45)).rotate([0,4,0])

# ax.scatter3D(rot_point[0], rot_point[1], rot_point[2], c='grey');


# plt.show()


# In[17]:


def getCameraPose(points, R, tvec, trans_multiplier=1.5):
    # takes in a list of any amount of points and rotates them
    def rotatePoints(points, R):
        points = points.transpose()
        rotated_points = []
        for point in points:
            rotated_points.append(np.array(point).dot(R))

        return np.array(rotated_points).transpose()
    
    def transposePoints(points, trans_vec):
        points = points.transpose()

        for i in range(points.shape[0]):
            points[i] = np.add(points[i], trans_vec.flatten())

        return points.transpose()
    
    points = rotatePoints(points, R)
    points = transposePoints(points, tvec*trans_multiplier)
    
    return points


# In[18]:


def drawCameraPose(cameraPoints, ax, index='0'):
    # this function draws black lines between cameraPoints
    # to make a camera looking thing 
    
    # 0 to 1 .all others
    for i in range(5):
        x_line = np.linspace(cameraPoints[0][0],cameraPoints[0][i],50)
        y_line = np.linspace(cameraPoints[1][0],cameraPoints[1][i],50)
        z_line = np.linspace(cameraPoints[2][0],cameraPoints[2][i],50)
        ax.plot3D(x_line, y_line, z_line,'black')
        
    for i in range(1,4):
        x_line = np.linspace(cameraPoints[0][i],cameraPoints[0][i+1],50)
        y_line = np.linspace(cameraPoints[1][i],cameraPoints[1][i+1],50)
        z_line = np.linspace(cameraPoints[2][i],cameraPoints[2][i+1],50)
        ax.plot3D(x_line, y_line, z_line,'black')
    
    x_line = np.linspace(cameraPoints[0][4],cameraPoints[0][1],50)
    y_line = np.linspace(cameraPoints[1][4],cameraPoints[1][1],50)
    z_line = np.linspace(cameraPoints[2][4],cameraPoints[2][1],50)
    
    # write the label
#     ax.scatter3D(cameraPoints[0][0], cameraPoints[1][0], cameraPoints[2][0], c='red');
    ax.text(cameraPoints[0][0], cameraPoints[1][0], cameraPoints[2][0], index, color='red')
    
    ax.plot3D(x_line, y_line, z_line,'black')


# In[19]:


from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

def drawArucoMarker(ax):
    '''
        This function draws a black rectangle at the origin
    '''
    # Draw a circle on the x=0 'wall'
    p = Rectangle((-3, 2), 6, 4)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")


# ### Rotate and translate cameraPoints

# In[20]:


# inline plots
# %matplotlib inline


# In[21]:


# windowed plots
# %matplotlib qt


# In[22]:


# draw a camera at the origin
# x_points = [0, 1, 1, 1, 1]
# y_points = [0, 0.5,-0.5,-0.5, 0.5]
# z_points = [0, 0.4, 0.4,-0.4,-0.4]
# cameraPoints = np.array([x_points,y_points,z_points])

# # cameraPoints = getCameraPose(cameraPoints,rot_vec, trans_vec)  # rotate and translate those camera points

# ax = create3DPlot()
# drawCameraPose(cameraPoints, ax)   # draw them in 3D
# plt.show()


# In[23]:


# drawArucoMarker(ax)
# plt.show()


# # Estimate Camera Pose for all left pictures

# In[24]:


dic = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

x_points = [0, 1, 1, 1, 1]
y_points = [0, 0.5,-0.5,-0.5, 0.5]
z_points = [0, 0.4, 0.4,-0.4,-0.4]


# smaller camera
x_points = [0, 0.5, 0.5, 0.5, 0.5]
y_points = [0, 0.25,-0.25,-0.25, 0.25]
z_points = [0, 0.2, 0.2,-0.2,-0.2]

og_cameraPoints = np.array([x_points,y_points,z_points])

ax = create3DPlot(bound=6)

verbose = True
# verbose = False
for i, image in enumerate(left_):
    markers = aruco.detectMarkers(image,dic)   # detect the marker on a single image
    
    objPoints = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],dtype=np.float64)
    
    # rotation vector and translation vector of the camera relative to the objectPoints
    retval, rot_vec, trans_vec = cv2.solvePnP(objPoints, markers[0][0][0], left_intrinsic, distCoeffs_left)
    
    if(verbose):
        print("\nleft_", i)
        print("R:\n", cv2.Rodrigues(rot_vec)[0])
        print("t:\n", trans_vec*3)
    
    cameraPoints = getCameraPose(og_cameraPoints.copy(),cv2.Rodrigues(rot_vec)[0], trans_vec, trans_multiplier=2)  # rotate and translate those camera points
    drawCameraPose(cameraPoints, ax, str(i))   # draw them in 3D
    
    if i == 6:
        break
plt.show()    


# In[ ]:





# In[ ]:




