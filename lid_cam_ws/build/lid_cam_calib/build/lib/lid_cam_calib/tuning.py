import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import matplotlib.image as mpimg
import cv2
import pdb

points = np.genfromtxt("/home/anthony/workspaces/lid_cam_ws/xyz.csv", delimiter=',', names=None)
img = cv2.imread('/home/anthony/workspaces/lid_cam_ws/cam2.png')
cv2.imshow('image', img)
cv2.waitKey(0)
    
def overlay(self, img,num, ret, pc_sensor, pc_centre):
# P2 (3 x 4) for left eye
    s = time.time()
    
    #velocity compensation using the gnss
    # v = [vgnss.x, vgnss.y,vgnss.z]
    # v = [inspva.north_velocity*np.cos(inspva.azimuth)-inspva.east_velocity*np.sin(inspva.azimuth) , inspva.north_velocity*(-1)*np.sin(inspva.azimuth)-inspva.east_velocity*np.cos(inspva.azimuth), 0]
    v = [0, 0/3.6, 0]
    # delta t between camera and lidar closest to it
    dtcam = (pc_sensor.header.stamp.sec + pc_sensor.header.stamp.nanosec/1e9) - (img.header.stamp.sec + img.header.stamp.nanosec/1e9)
    # delta t for lidar closest to camera and centre lidar
    if pc_centre != 0:
        dtlidar = (pc_centre.header.stamp.sec + pc_centre.header.stamp.nanosec/1e9) - (pc_sensor.header.stamp.sec + pc_sensor.header.stamp.nanosec/1e9)
    else:
        dtlidar = 0
    # dt = 0
    print("num: ", num, " v: ", v, " dt: ", dtcam)

    pc, Tr_cepton_to_cam, projection = self.properties(num, pc_sensor, v, dtcam, dtlidar)       

    # cepton: 0: RIGHT, 1: FWD, 2: UP
    # cepton -> velo 

    cepton_pts = pc.T

    # Calculate distances
    # Calculate the Euclidean distance along the third axis (axis=0)
    distance_vector = np.linalg.norm(cepton_pts, axis=0)
    
    # Set the threshold distance
    # Remove points outside the scoring + 5m radius (40 + 5m)
    threshold = 45.0
    # # Filter out the points that exceed the threshold
    filtered_pts = cepton_pts[:, distance_vector < threshold]
    # Insert a column of ones
    cepton = np.insert(filtered_pts.T,3,1,axis=1).T
    
    #adjusting for different coordinate frames between the cameras and lidar
    #apply all transformations to each other first, then to the 
    tm_proj_cam_xy, transformation_matrix_cam_xy = self.trans_matrix(Tr_cepton_to_cam, projection)

    #cam is an array of {u,v,z}, where u,v are pixels, and z is depth
    cam = tm_proj_cam_xy.dot(cepton)

    cepton = np.delete(cepton,np.where(cam[2,:]<=0),axis=1)
    cam = np.delete(cam,np.where(cam[2,:]<=0),axis=1)
    # # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    IMG_H = 1536
    IMG_W = 2048
    
    # filter point out of canvas
    uu = cam[0]
    vv = cam[1]
    zz = cam[2]

    #crops points that are not contained in the image frame
    u_out = np.logical_or(uu<0, uu>IMG_W)
    v_out = np.logical_or(vv<0, vv>IMG_H)
    #list of all points that are not in the frame
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)

    #get corresponding xyz for uvz -> 
    #from [u,v,z] obtain [u,v,{x,y,z}], matched by index to point cloud
    cepton = (transformation_matrix_cam_xy).dot(cepton)
    # cepton = mat_mult(transformation_matrix_cam_xy, cepton)
    cepton = np.delete(cepton,np.where(outlier),axis=1)

    #store into custom data type
    xyz = np.zeros((IMG_H+1, IMG_W+1, 3), dtype=np.float16)
    cam = np.round(cam).tolist()
    cam = np.array(cam).astype(np.uint16)
    a = list(zip(cepton[0], cepton[1], cepton[2]))
    length = len(cam[1])
    a = a[:length]
    # print("a: ", len(a),"cam[1]: ", cam[1].shape, "cam[0]: ", cam[0].shape)
    xyz[cam[1], cam[0]] = a
    xyz = xyz[:-1, :-1, :]
    ret[num] = xyz
    # my_global_array[num] = xyz
    # print(np.sum(xyz))
    
    imagecv2 = bridge.imgmsg_to_cv2(img, desired_encoding='rgb8')
    # save data
    folder = "union-albert"
    # self.data_collect(imagecv2, xyz, num, folder)

    # show overlay
    flag = 0
    # self.visualization(imagecv2, cam, num, flag, cepton)

def merge_one_lidar(self, cloud_i, theta_i, ti_x, ti_y, ti_z):
    pcd_npi = np.array(list(read_points(cloud_i)))
    xi = pcd_npi[:,0:1]
    yi = pcd_npi[:,1:2]
    zi = pcd_npi[:,2:3]
    pointsi = np.column_stack((xi, yi, zi))
    Tr_AtoA = np.array([[np.cos(theta_i),-np.sin(theta_i),0,ti_x],
                [np.sin(theta_i), np.cos(theta_i),0,        ti_y],
                [0            ,0             ,1,            ti_z]])

    if(theta_i != 0):
        pointsi = pointsi.dot(Tr_AtoA)
        # pointsi = mat_mult(pointsi, Tr_AtoA)
    else:
        pointsi = np.insert(pointsi.T,3,1,axis=1).T

    
    return pointsi

def properties(self, num, pc_sensor, velocity, dtcam, dtlidar):
    if(num==1):
        theta = np.pi/6
        tx = -0.17   + velocity[0]*dtcam    #right
        tz = 0.009   + velocity[1]*dtcam  # forward   
        ty = -0.0015 + velocity[2]*dtcam     #-up
        projection = np.array([1100.727661, 0.000000, 1028.897396, 0.000000, 1161.720093, 771.434530, 0.000000, 0.000000, 1.000000]).reshape(3,3)

        theta_i = np.pi/6   
        ti_x = -0.255 + velocity[0]*dtlidar    #right
        ti_y = 0      + velocity[1]*dtlidar      #up
        ti_z = -0.061 + velocity[2]*dtlidar   # forward
        pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
        # pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
        # pc = np.vstack((pcL[:,:3],pcC[:,:3]))
    elif(num==2):
        theta = 0
        tx = -0.111 + velocity[0]*dtcam 
        ty = -0.0015  + velocity[2]*dtcam 
        tz = -0.007  + velocity[1]*dtcam
        projection = np.array([1127.220459, 0.000000, 1030.718004, 0.000000, 1184.828491, 749.131132, 0.000000, 0.000000, 1.000000]).reshape(3,3)

        theta_i = 0
        ti_x = 0    #right
        ti_y = 0      #up
        ti_z = 0   # forward
        pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
    elif(num==3):
        theta = 0
        tx = -0.111 + velocity[0]*dtcam 
        ty = -0.0015  + velocity[2]*dtcam 
        tz = -0.007 + velocity[1]*dtcam
        projection = np.array([1131.871948, 0.000000, 1022.029700, 0.000000, 1189.723755, 766.790667, 0.000000, 0.000000, 1.000000]).reshape(3,3)

        theta_i = 0     
        ti_x = 0    #right
        ti_y = 0      #up
        ti_z = 0   # forward
        pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
    elif(num==4):
        theta = -np.pi/6
        tx = 0.17 + velocity[0]*dtcam
        ty = -0.0015 + velocity[2]*dtcam  
        tz = -0.007 + velocity[1]*dtcam
        projection = np.array([1124.364502, 0.000000, 1004.415648, 0.000000, 1185.257812, 767.772319, 0.000000, 0.000000, 1.000000]).reshape(3,3)

        theta_i = np.pi/6     
        ti_x = 0.255 + velocity[0]*dtlidar    #right
        ti_y = 0     + velocity[1]*dtlidar  #up
        ti_z = -0.061+ velocity[2]*dtlidar   # forward
        pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
        # pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
        # pc = np.vstack((pcR[:,:3],pcC[:,:3]))

    
    Tr_cepton_to_cam = np.array([[np.cos(theta),0,np.sin(theta), tx],
                                [       0,      1,      0,       ty],
                                [-np.sin(theta),0,np.cos(theta),tz]])

    return pc, Tr_cepton_to_cam, projection

def trans_matrix(self, Tr_cepton_to_cam, projection):
    theta = 0
    transformation_matrix_y = np.array([[np.cos(theta)  ,0  ,np.sin(theta)],
                                    [0, 1,0],
                                    [-np.sin(theta), 0, np.cos(theta)]])

    theta = + np.pi/2 
    transformation_matrix_x = np.array([[1            ,0             ,0],
                                    [0, np.cos(theta),-np.sin(theta)],
                                    [0, np.sin(theta), np.cos(theta)]])

    transformation_matrix_xy = np.dot(transformation_matrix_x, transformation_matrix_y)
    # cepton_pts = transformation_matrix_xy.dot(cepton_pts)


    column_to_be_added = np.array([[0], [0], [0]])

    # Adding column to array using append() method
    transformation_matrix_xy = np.append(transformation_matrix_xy, column_to_be_added, axis=1)
    transformation_matrix_xy = np.vstack([transformation_matrix_xy, [0, 0, 0, 1]])
    transformation_matrix_cam_xy = Tr_cepton_to_cam.dot(transformation_matrix_xy)
    tm_proj_cam_xy = projection.dot(transformation_matrix_cam_xy)

    return tm_proj_cam_xy, transformation_matrix_cam_xy

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set(xlim = (-25,25), ylim = (-45,5), zlim = (-25,25))
ax.scatter3D(-points[:,0:1],-points[:,2:3],-points[:,1:2],color='blue', depthshade=True, s=0.01);
# ax.scatter3D(points2[:,0:1],points2[:,1:2],points2[:,2:3],color='red', depthshade=True, s=0.01);
ax.scatter3D(0,0,0, s=20)
ax.view_init(elev=5,azim=90);
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.show()
plt.close()