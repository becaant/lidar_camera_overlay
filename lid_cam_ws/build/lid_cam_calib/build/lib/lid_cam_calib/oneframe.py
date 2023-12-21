import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import matplotlib.image as mpimg
import cv2
import pdb
from sklearn import preprocessing

# # Load data from a CSV file with headers
# x = np.genfromtxt("x.csv", delimiter=',', names=None)
# y = np.genfromtxt("y.csv", delimiter=',', names=None)
# z = np.genfromtxt("z.csv", delimiter=',', names=None)
# intensity = np.genfromtxt("intensity.csv", delimiter=',', names=None)
# scatter = plt.scatter(x, y, z, c=intensity, cmap='hot')
from sklearn.preprocessing import MinMaxScaler

# Load the data from CSV files
x = np.genfromtxt("x.csv", delimiter=',', names=None)
y = np.genfromtxt("y.csv", delimiter=',', names=None)
z = np.genfromtxt("z.csv", delimiter=',', names=None)
intensity = np.genfromtxt("intensity.csv", delimiter=',', names=None)

valid_indices = np.where((x >= -1) & (x <= 1) & (y < 3.5))
x_filtered = x[valid_indices]
y_filtered = y[valid_indices]
z_filtered = z[valid_indices]
intensity_filtered = intensity[valid_indices]

# Normalize the intensity values
scaler = MinMaxScaler()
intensity_normalized = scaler.fit_transform(intensity_filtered.reshape(-1, 1)).flatten()

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Display the 3D point cloud with normalized intensity values as a color map
scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=intensity_normalized, cmap='hot')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Add a colorbar to show the mapping of intensity values to colors
cbar = plt.colorbar(scatter)

# Show the plot
plt.show()

x = x_filtered
y = y_filtered
z = z_filtered
intensity = intensity_filtered

# pdb.set_trace()


if (len(x)<len(y)):
    if (len(x)<len(z)):
        length = len(x)
    elif (len(y)<len(z)):
        length = len(y)
else:
    length = len(z)
# pdb.set_trace()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set(xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
# ax.scatter3D(x[:length],y[:length],z[:length],c=None, depthshade=True, s=0.01);
# ax.scatter3D(0,0,0, s=20)
# ax.view_init(elev=5,azim=90);
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')
#plt.show()
#X is right, Y is forward, Z is up


#plt.close()

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

tx = 0.08    #right
ty = -0.0015      #-up
tz = 0.037   # forward
Tr_cepton_to_cam = np.array([[np.cos(theta),0,np.sin(theta), tx],
                             [0,            1,0,             ty],
                             [-np.sin(theta),0,np.cos(theta),tz]])


points = np.column_stack((x, y, z))

name = "oneframe"
# TODO: use fov filter? 
#cepton = np.delete(cepton,np.where(cepton[1,:]<0),axis=1)
# velo: 0: FWD, 1: LEFT, 2: UP
# cepton: 0: RIGHT, 1: FWD, 2: UP
# cepton -> velo 
transformation_matrix = np.array([[0, 0, 1],
                                [1, 0, 0],
                                [0, -1, 0]])
theta =  0 # FOR TROUBLESHOOTING

transformation_matrix_z = np.array([[np.cos(theta),-np.sin(theta),0],
                                  [np.sin(theta), np.cos(theta),0],
                                  [0            ,0             ,1]])

cepton_pts = np.dot(transformation_matrix.T, points.T)

cepton_pts = points.T

theta = 0
transformation_matrix_y = np.array([[np.cos(theta)  ,0  ,np.sin(theta)],
                                  [0, 1,0],
                                  [-np.sin(theta), 0, np.cos(theta)]])

cepton_pts = np.dot(transformation_matrix_y, cepton_pts)

theta = + np.pi/2 
transformation_matrix_x = np.array([[1            ,0             ,0],
                                  [0, np.cos(theta),-np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]])

cepton_pts = np.dot(transformation_matrix_x, cepton_pts)
cepton = np.insert(cepton_pts.T,3,1,axis=1).T
cam = P2.dot(Tr_cepton_to_cam.dot(cepton))
cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)
# # get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
# filter point out of canvas

uu = cam[0]
vv = cam[1]
zz = cam[2]
#pdb.set_trace()
u_out = np.logical_or(uu<0, uu>IMG_W)
v_out = np.logical_or(vv<0, vv>IMG_H)
outlier = np.logical_or(u_out, v_out)
#pdb.set_trace()
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
# pdb.set_trace()
count = 0

pdb.set_trace()
for i in intensity:
    if i > 5:
        intensity[count] = 0

        # pdb.set_trace()
    count = count + 1


colors= preprocessing.normalize([intensity])
# colors= [intensity]

# pdb.set_trace()
plt.scatter([uu],[vv],c=colors,cmap='rainbow_r',s=20)
# Display the 3D point cloud with intensity values as a color map

plt.title(name)
plt.savefig("calib.png")
#plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
plt.show()