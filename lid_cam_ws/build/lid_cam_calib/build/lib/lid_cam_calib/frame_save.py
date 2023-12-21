import multiprocessing
import time
# from ros_perception import PerceptionNode
import rclpy
from rclpy.node import Node
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import numpy as np
# import mk
import pdb
from cv_bridge import CvBridge

import sys
import math
import struct
import matplotlib.pyplot as plt
import matplotlib as cm
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
import threading
import pickle
# from numba import njit, prange
# from main_node import PerceptionNode
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

# Custom Message imports
from geometry_msgs.msg import Point

# mkl.set_num_threads(1)

bridge = CvBridge()
plt.ion()

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

# my_global_array = [0,0,0,0]

# @njit(parallel=True)
# def mat_mult(A, B):
    # assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )

    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res

# @njit
# def mat_mult(A, B):

#     return np.matmul(A, B)



class RosPerceptionNode(Node):

    def __init__(self):
        super().__init__('ros_perception_node')
        # self.subscription = self.create_subscription(Image,'/lucid_vision/camera_1/image',self.im_callback,10)
        
        #bus
        # self.image2 = Subscriber(self, Image,"/lucid_vision/camera_1/image")
        # self.image2 = Subscriber(self, Image,"/lucid_vision/camera_2/image")
        # self.image3 = Subscriber(self, Image,"/lucid_vision/camera_3/image")
        # self.image4 = Subscriber(self, Image,"/lucid_vision/camera_4/image")

        self.image1 = Subscriber(self, Image,"/lucid_vision/camera_1/image_rect")
        self.image2 = Subscriber(self, Image,"/lucid_vision/camera_2/image_rect")
        self.image3 = Subscriber(self, Image,"/lucid_vision/camera_3/image_rect")
        self.image4 = Subscriber(self, Image,"/lucid_vision/camera_4/image_rect")
        
        # self.centre = Subscriber(self, PointCloud2, "center_lidar")  # X90 (Center)
        #bus
        self.centre = Subscriber(self, PointCloud2, "center_lidar")  # X90 (Center)
        self.left = Subscriber(self, PointCloud2, "left_lidar")   # Left  P60 
        self.right = Subscriber(self, PointCloud2, "right_lidar")   # Right P60

        # self.counts = [0,0,0,0]
        # self.fps = [0,0,0,0]

        #IP for lidars is 192.168.x.x netmask 16
        qos = 1
        slop = 5
        #bus
        # self.tsa = ApproximateTimeSynchronizer([self.image2, self.image3, self.centre],qos, slop)
        
        self.ts1 = ApproximateTimeSynchronizer([self.image1, self.image2, self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts2 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts3 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts4 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)

        
        self.globaltime = float(self.get_clock().now().nanoseconds)/10**9
        # print("globaltime", self.globaltime)
        # self.tsa.registerCallback(self.cba)

        #bus 
        self.ts1.registerCallback(self.cb1)
        self.ts2.registerCallback(self.cb2)
        self.ts3.registerCallback(self.cb3)
        self.ts4.registerCallback(self.cb4)
        # self.perception_node = PerceptionNode()
    
    s = time.time()
            
    def cba(self, img2,img3, pcCenter):      
        threads_results = {}
        print("hello")
        s = time.time()
        self.overlay(img2,2,threads_results, pcCenter, 0)
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e = time.time()
        fps = 1/(e-s)
        print("Lidars to cam 2 at ", fps, "")
        # exit()
        
        
    def cb1(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):      
        threads_results = {}

        self.overlay(img1,1,threads_results, pcCenter,pcLeft)
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})

        print("Lidars to cam 1")
        # exit()
        
    
    def cb2(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        threads_results = {}

        self.overlay(img2,2,threads_results, pcCenter,0)
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        print("Lidars to cam 2")

    
    def cb3(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        threads_results = {}
        self.overlay(img3,3,threads_results, pcCenter,0)
        
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e2 = time.time()
        print("Lidars to cam 3")
        
    
    def cb4(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        threads_results = {}
        self.overlay(img4,4, threads_results, pcRight,pcCenter) 
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        print("Lidars to cam 4")
        
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
        # print("num: ", num, " v: ", v, " dt: ", dtcam)

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
        if cepton.shape[0]==5:
            cepton = np.delete(cepton, 4, axis=0)

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
        self.visualization(imagecv2, cam, num, flag, cepton)
    
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
            theta = np.pi/72
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
            theta = -np.pi/72
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
            theta = -np.pi/72
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
            theta = -np.pi/6 - np.pi/36
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

    def visualization(self, img, cam, num, flag, pc):
        if num != 0:
                cv2.imwrite('/home/anthony/lidar_camera_overlay/lid_cam_ws/rect_img.png', img)
                plt.axis([0,2048,1536,0])
                plt.imshow(img)
                plt.scatter(cam[0],cam[1],c=cam[2],cmap='gist_rainbow',alpha=0.5,s=.5)
                plt.colorbar()

                if flag == 0:   
                    #save
                    plt.ioff()
                    plt.show()
                    plt.savefig(f'/home/anthony/lidar_camera_overlay/lid_cam_ws/cam' + str(num) + '.png')
                    plt.close()
                    # file_path = "/home/anthony/workspaces/lid_cam_ws/xyz.csv"
                    # np.savetxt(file_path, pc.T, delimiter=',', header='X,Y,Z', comments='')
                else:
                    #show
                    plt.show() 
                    plt.close()

                    
        # cam_computed = np.sqrt(cepton[0][:]**2+cepton[1][:]**2+cepton[2][:]**2) 
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].scatter([i for i in range(len(cam[2]))], cam[2], c = 'r', s=5)
        # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed[:len(cam[2])], c = 'b', s=5)
        # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed - cam[2], c = 'y', s=5)


def main(args=None):
    rclpy.init(args=args)
    ros_perception_node = RosPerceptionNode()

    rclpy.spin(ros_perception_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
# transformation_matrix_x