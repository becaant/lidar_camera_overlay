import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import matplotlib.image as mpimg
import cv2
import pdb

def pythagorean(lidar, i, cam):
    print(np.sqrt(lidar[0][i]**2+lidar[1][i]**2+lidar[2][i]**2), cam[2][i])

# Load data from a CSV file with headers
x = np.genfromtxt("x.csv", delimiter=',', names=None)
y = np.genfromtxt("y.csv", delimiter=',', names=None)
z = np.genfromtxt("z.csv", delimiter=',', names=None)

if (len(x)<len(y)):
    if (len(x)<len(z)):
        length = len(x)
    elif (len(y)<len(z)):
        length = len(y)
else:
    length = len(z)

length = len(y)
# pdb.set_trace()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set(xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
# ax.scatter3D(x[:length],y[:length],z[:length],c=None, depthshade=True, s=0.01);
# ax.scatter3D(0,0,0, s=20)
# ax.view_init(elev=5,azim=90);
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')
#plt.zlabel('Z [m]')
#plt.show()
#X is right, Y is forward, Z is up


# #plt.close()

# img = plt.imread("undistorted1.png")
img = f'./undistorted1.png'
# binary = f'./data_object_velodyne/testing/velodyne/{name}.bin'
# with open(f'./testing/calib/{name}.txt','r') as f:
#     calib = f.readlines()

# P2 (3 x 4) for left eye
P2 = np.array([1100.727661, 0.000000, 1028.897396, 
                0.000000, 1161.720093, 771.434530, 
                0.000000, 0.000000, 1.000000]).reshape(3,3)

R0_rect = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
# Add a 1 in bottom-right, reshape to 4 x 4
R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
#print("dot", P2.dot(R0_rect))

theta = 0
tx = -0.075    #right
ty = -.0015      #up
tz = .042   # forward
Tr_cepton_to_cam = np.array([[np.cos(theta),-np.sin(theta),0,tx],
                             [np.sin(theta), np.cos(theta),0,ty],
                             [0            ,0             ,1,tz]])

# Tr_cepton_to_cam = np.insert(Tr_cepton_to_cam,3,values=[0,0,0,1],axis=0)
#print(Tr_cepton_to_cam)
# read raw data from binary
# scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
# Stack the arrays into one array
points = np.column_stack((x[:length], y[:length], z[:length]))
#pdb.set_trace()
# Print the result
#print(points)
name = "oneframe"
# TODO: use fov filter? 
#cepton = np.delete(cepton,np.where(cepton[1,:]<0),axis=1)
# velo: 0: FWD, 1: LEFT, 2: UP
# cepton: 0: RIGHT, 1: FWD, 2: UP
# cepton -> velo 
transformation_matrix = np.array([[0, 0, 1],
                                [1, 0, 0],
                                [0, -1, 0]])
theta =  0

transformation_matrix_z = np.array([[np.cos(theta),-np.sin(theta),0],
                                  [np.sin(theta), np.cos(theta),0],
                                  [0            ,0             ,1]])
#pdb.set_trace()
cepton_pts = np.dot(transformation_matrix.T, points.T)

#cepton_pts = np.dot(transformation_matrix_z, points.T)
cepton_pts = points.T

theta = 0
transformation_matrix_y = np.array([[np.cos(theta)  ,0  ,np.sin(theta)],
                                  [0, 1,0],
                                  [-np.sin(theta), 0, np.cos(theta)]])

theta = + np.pi/2
transformation_matrix_x = np.array([[1            ,0             ,0],
                                  [0, np.cos(theta),-np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]])

cepton_pts = np.dot(transformation_matrix_x, cepton_pts)
# cepton_pts = cepton_pts.T
cepton = np.insert(cepton_pts.T,3,1,axis=1).T

#cam is an array of {x,y,z}, where x,y are pixels, and z is depth
cam = P2.dot(Tr_cepton_to_cam.dot(cepton))
cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
# # get u,v,z
# #pdb.set_trace()
cam[:2] /= cam[2,:]
# do projection staff
# plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range

# filter point out of canvas

uu = cam[0]
vv = cam[1]
zz = cam[2]
pdb.set_trace()

#crops points that are not contained in the image frame
u_out = np.logical_or(uu<0, uu>IMG_W)
v_out = np.logical_or(vv<0, vv>IMG_H)
#list of all points that are not in the frame
outlier = np.logical_or(u_out, v_out)

# pdb.set_trace()
#removes outliers
cam = np.delete(cam,np.where(outlier),axis=1)

# test = np.delete(cepton, 3,0)
test = cepton
test = (Tr_cepton_to_cam).dot(test)
# pdb.set_trace()
test = np.delete(test,np.where(outlier),axis=1)
cepton = np.delete(cepton,np.where(outlier),axis=1)

# generate color map from depth
#u,v,z = cam
# lidar_pts = np.linalg.solve(P2.dot(Tr_cepton_to_cam), cam)


cam_computed = np.sqrt(test[0][:]**2+test[1][:]**2+test[2][:]**2) 
cam_computed2 = np.sqrt(cepton[0][:]**2+cepton[1][:]**2+cepton[2][:]**2) 

# xyz = np.zeros((IMG_H+1, IMG_W+1, 3), dtype="uint")
# cam = np.round(cam).tolist()
# cam = np.array(cam).astype(np.uint32)
# a = list(zip(test[0], test[1], test[2]))

# xyz[cam[1], cam[0]] = a
# xyz = xyz[:-1, :-1, :]
# print(xyz.shape)

pdb.set_trace()

# import pickle
# with open("path.txt", "wb") as fw:
#     pickle.save(fw, xyz)
    
output = (np.row_stack((np.round(cam[0]), np.round(cam[1]), test[0], test[1], test[2]))).T
#                           u                   v               x       y       z

# pdb.set_trace()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter([i for i in range(len(cam[2]))], cam[2], c = 'r', s=5)
axs[0].scatter([i for i in range(len(cam[2]))], cam_computed, c = 'b', s=5)
axs[0].scatter([i for i in range(len(cam[2]))], cam_computed2, c = 'g', s=5)
axs[0].scatter([i for i in range(len(cam[2]))], cam_computed - cam[2], c = 'y', s=5)
axs[0].scatter([i for i in range(len(cam[2]))], cam_computed2 - cam[2], c = 'k', s=5)

# print(np.sum(cam_computed2))

axs[1].axis([0,IMG_W,IMG_H,0])
axs[1].imshow(png)
scatter = axs[1].scatter([uu],[vv],c=[zz],cmap='rainbow_r',alpha=0.5,s=10)
cbar = fig.colorbar(scatter, ax=axs[1])
#plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
plt.tight_layout()
plt.show()


# Point.msg
#     float x
#     float y
#     float z

# Calibration.msg
#     Point[][] calibration