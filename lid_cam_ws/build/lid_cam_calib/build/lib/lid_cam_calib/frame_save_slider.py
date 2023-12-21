import multiprocessing
import time
# from ros_perception import PerceptionNode
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
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
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import threading
import pickle
from numba import njit, prange
# from main_node import PerceptionNode
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

# Custom Message imports
from geometry_msgs.msg import Point

# mkl.set_num_threads(1)
slider_tx=0.0
slider_ty=0.0
slider_tz=0.0
slider_theta=0.0

img=None
num=None
ret=None
pc_sensor=None
pc_centre=None


bridge = CvBridge()

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

@njit(parallel=True)
def mat_mult(A, B):
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


def on_change_x(num):
    global slider_tx, slider_ty, slider_tz, slider_theta

    # imageCopy = img.copy()
    slider_tx=num
    overlay2()
    imageCopy = cv2.imread('/home/autodrive/Desktop/test/cam1.png')
    cv2.imshow('tuning', imageCopy)

def on_change_y(num):
    global slider_tx, slider_ty, slider_tz, slider_theta
  
    # imageCopy = img.copy()
    slider_ty=num
    overlay2()
    imageCopy = cv2.imread('/home/autodrive/Desktop/test/cam1.png')

    cv2.imshow('tuning', imageCopy)

def on_change_z(num):
    global slider_tx, slider_ty, slider_tz, slider_theta
   
    # imageCopy = img.copy()
    slider_tz=num
    overlay2()
    imageCopy = cv2.imread('/home/autodrive/Desktop/test/cam1.png')
    cv2.imshow('tuning', imageCopy)

def on_change_th(num):
    global slider_tx, slider_ty, slider_tz, slider_theta

    # imageCopy = img.copy()
    slider_theta=num
    overlay2()
    imageCopy = cv2.imread('/home/autodrive/Desktop/test/cam1.png')
    cv2.imshow('tuning', imageCopy)



class RosPerceptionNode(Node):

    def __init__(self):
        super().__init__('ros_perception_node')
        # self.subscription = self.create_subscription(Image,'/lucid_vision/camera_1/image',self.im_callback,10)
        self.image1 = Subscriber(self, Image,"/lucid_vision/camera_1/image_rect")
        self.image2 = Subscriber(self, Image,"/lucid_vision/camera_2/image_rect")
        self.image3 = Subscriber(self, Image,"/lucid_vision/camera_3/image_rect")
        self.image4 = Subscriber(self, Image,"/lucid_vision/camera_4/image_rect")

        self.centre = Subscriber(self, PointCloud2, "center_lidar")  # X90 (Center)
        self.left = Subscriber(self, PointCloud2, "left_lidar")   # Left  P60 
        self.right = Subscriber(self, PointCloud2, "right_lidar")   # Right P60

        # self.counts = [0,0,0,0]
        # self.fps = [0,0,0,0]

        #IP for lidars is 192.168.x.x netmask 16
        qos = 5
        slop = 2
        self.ts1 = ApproximateTimeSynchronizer([self.image1, self.image2, self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts2 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts3 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)
        self.ts4 = ApproximateTimeSynchronizer([self.image1, self.image2,self.image3, self.image4, self.centre, self.left, self.right],qos, slop)

        
        self.globaltime = float(self.get_clock().now().nanoseconds)/10**9
        # print("globaltime", self.globaltime)
        self.ts1.registerCallback(self.cb1)
        self.ts2.registerCallback(self.cb2)
        self.ts3.registerCallback(self.cb3)
        self.ts4.registerCallback(self.cb4)
        # self.perception_node = PerceptionNode()
    
    s = time.time()
        
    def cb1(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        e = time.time()
        s1 = time.time()
        
        e1 = time.time()
        s2 = time.time() 
        
        threads_results = {}

        self.overlay(img1,1,threads_results, pcCenter,pcLeft)

        # # img1 = bridge.imgmsg_to_cv2(img1, desired_encoding='rgb8')
        # img2 = bridge.imgmsg_to_cv2(img2, desired_encoding='rgb8')
        # img3 = bridge.imgmsg_to_cv2(img3, desired_encoding='rgb8')
        # img4 = bridge.imgmsg_to_cv2(img4, desired_encoding='rgb8')
        
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e2 = time.time()
        # self.count[0] +=1
        # self.fps[0] += np.round(1/(e2-s1))
        print("Lidars to cam 1", np.round(e2-s2, 4),"FPS ", np.round(1/(e2-s1), 1))
        # exit()
        
    
    def cb2(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        s1 = time.time()
        e1 = time.time()
        s2 = time.time() 
        
        threads_results = {}

        # self.overlay(img1,1, threads_results,  pcLeft,pcCenter)
        self.overlay(img2,2,threads_results, pcCenter,0)
        # self.overlay(img3,3,threads_results, pcCenter,0)
        # self.overlay(img4,4, threads_results, pcRight,pcCenter)

        # # img1 = bridge.imgmsg_to_cv2(img1, desired_encoding='rgb8')
        # img2 = bridge.imgmsg_to_cv2(img2, desired_encoding='rgb8')
        # img3 = bridge.imgmsg_to_cv2(img3, desired_encoding='rgb8')
        # img4 = bridge.imgmsg_to_cv2(img4, desired_encoding='rgb8')
        
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e2 = time.time()
        print("Lidars to cam 2", np.round(e2-s2, 4),"FPS ", np.round(1/(e2-s1), 1))

        # exit()
         
    
    def cb3(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        e = time.time()
        # print('hello')
        s1 = time.time()
        
        # points = self.lidar_merge(pcCenter,pcLeft,pcRight)
        e1 = time.time()
        s2 = time.time() 
        
        threads_results = {}

        # self.overlay(img1,1, threads_results,  pcLeft,pcCenter)
        # self.overlay(img2,2,threads_results, pcCenter,0)
        self.overlay(img3,3,threads_results, pcCenter,0)
        # self.overlay(img4,4, threads_results, pcRight,pcCenter)

        # # img1 = bridge.imgmsg_to_cv2(img1, desired_encoding='rgb8')
        # img2 = bridge.imgmsg_to_cv2(img2, desired_encoding='rgb8')
        # img3 = bridge.imgmsg_to_cv2(img3, desired_encoding='rgb8')
        # img4 = bridge.imgmsg_to_cv2(img4, desired_encoding='rgb8')
        
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e2 = time.time()
        print("Lidars to cam 3", np.round(e2-s2, 4),"FPS ", np.round(1/(e2-s1), 1))

        # exit()
        
    
    def cb4(self, img1,img2,img3,img4, pcCenter,pcLeft,pcRight):
        e = time.time()
        # print('hello')
        s1 = time.time()
        
        # points = self.lidar_merge(pcCenter,pcLeft,pcRight)
        e1 = time.time()
        s2 = time.time() 
        
        threads_results = {}

        # self.overlay(img1,1, threads_results,  pcLeft,pcCenter)
        # self.overlay(img2,2,threads_results, pcCenter,0)
        # self.overlay(img3,3,threads_results, pcCenter,0)
        self.overlay(img4,4, threads_results, pcRight,pcCenter)

        # # img1 = bridge.imgmsg_to_cv2(img1, desired_encoding='rgb8')
        # img2 = bridge.imgmsg_to_cv2(img2, desired_encoding='rgb8')
        # img3 = bridge.imgmsg_to_cv2(img3, desired_encoding='rgb8')
        # img4 = bridge.imgmsg_to_cv2(img4, desired_encoding='rgb8')
        
        # self.perception_node.process_frame({1:(img1, threads_results[1]), 2 :(img2, threads_results[2]), 
        #                                     3: (img3, threads_results[3]), 4 : (img4, threads_results[4])})
        e2 = time.time()
        print("Lidars to cam 4", np.round(e2-s2, 4),"FPS ", np.round(1/(e2-s1), 1))

        # exit()


        
    def overlay(self, _img,_num, _ret, _pc_sensor, _pc_centre):
        global slider_tx, slider_ty, slider_tz, slider_theta, slider_tx, slider_ty, slider_tz, slider_theta
        img,num, ret, pc_sensor, pc_centre =  _img,_num, _ret, _pc_sensor, _pc_centre

        # P2 (3 x 4) for left eye
        s = time.time()
        #get projection matrix
        # print(pc.shape)
        R0_rect = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
        # Add a 1 in bottom-right, reshape to 4 x 4
        R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
        R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)

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

        if(num==1):
            theta = np.pi/6          + slider_theta/1000
            tx = -0.17 + v[0]*dtcam  + slider_tx/1000   #right
            ty = -0.0015 + v[2]*dtcam+ slider_ty/1000   #-up
            tz = 0.009 + v[1]*dtcam  + slider_tz/1000 # forward   
            projection = np.array([1100.727661, 0.000000, 1028.897396, 0.000000, 1161.720093, 771.434530, 0.000000, 0.000000, 1.000000]).reshape(3,3)

            theta_i = 0   
            ti_x = -0.255 + v[0]*dtlidar    #right
            ti_y = 0 + v[0]*dtlidar      #up
            ti_z = -0.061 + v[0]*dtlidar   # forward
            # pcL = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
            # pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
            # pc = np.vstack((pcL[:,:3],pcC[:,:3]))

            #anthony troubleshooting
            pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
            # pc = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)

        elif(num==2):
            theta = 0+ slider_theta/1000
            tx = -0.111 + v[0]*dtcam + slider_tx/1000   
            ty = -0.0015  + v[2]*dtcam + slider_ty/1000   
            tz = -0.007  + v[1]*dtcam+ slider_tz/1000 
            projection = np.array([1127.220459, 0.000000, 1030.718004, 0.000000, 1184.828491, 749.131132, 0.000000, 0.000000, 1.000000]).reshape(3,3)

            theta_i = 0
            ti_x = 0    #right
            ti_y = 0      #up
            ti_z = 0   # forward
            pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
        elif(num==3):
            theta = -np.pi/72+ slider_theta/1000
            tx = 0.111 + v[0]*dtcam + slider_tx  /1000 
            ty = -0.0015  + v[2]*dtcam + slider_ty  /1000 
            tz = -0.007 + v[1]*dtcam+ slider_tz/1000 # 
            projection = np.array([1131.871948, 0.000000, 1022.029700, 0.000000, 1189.723755, 766.790667, 0.000000, 0.000000, 1.000000]).reshape(3,3)

            theta_i = 0     
            ti_x = 0    #right
            ti_y = 0      #up
            ti_z = 0   # forward
            pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
        elif(num==4):
            theta = -np.pi/6+ slider_theta/1000
            tx = 0.17 + v[0]*dtcam+ slider_tx /1000  
            ty = -0.0015 + v[2]*dtcam  + slider_ty  /1000 
            tz = 0.009 + v[1]*dtcam+ slider_tz/1000 # 
            projection = np.array([1124.364502, 0.000000, 1004.415648, 0.000000, 1185.257812, 767.772319, 0.000000, 0.000000, 1.000000]).reshape(3,3)

            theta_i = np.pi/6     
            ti_x = 0.255    #right
            ti_y = 0      #up
            ti_z = -0.061   # forward
            pcR = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
            pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
            pc = np.vstack((pcR[:,:3],pcC[:,:3]))


        Tr_cepton_to_cam = np.array([[np.cos(theta),0,np.sin(theta), tx],
                                    [0,            1,0,             ty],
                                    [-np.sin(theta),0,np.cos(theta),tz]])

        # velo: 0: FWD, 1: LEFT, 2: UP
        # cepton: 0: RIGHT, 1: FWD, 2: UP
        # cepton -> velo 

        cepton_pts = pc.T

        theta = 0
        transformation_matrix_y = np.array([[np.cos(theta)  ,0  ,np.sin(theta)],
                                        [0, 1,0],
                                        [-np.sin(theta), 0, np.cos(theta)]])

        # cepton_pts = np.dot(transformation_matrix_y, cepton_pts)

        theta = + np.pi/2 
        transformation_matrix_x = np.array([[1            ,0             ,0],
                                        [0, np.cos(theta),-np.sin(theta)],
                                        [0, np.sin(theta), np.cos(theta)]])

        transformation_matrix_xy = np.dot(transformation_matrix_x, transformation_matrix_y)
        # cepton_pts = transformation_matrix_xy.dot(cepton_pts)

        # Calculate distances
        # Calculate the Euclidean distance along the third axis (axis=0)
        distance_vector = np.linalg.norm(cepton_pts, axis=0)
        # # Set the threshold distance
        # # Remove points outside the scoring + 5m radius (40 + 5m)
        threshold = 45.0

        # # Filter out the points that exceed the threshold
        filtered_pts = cepton_pts[:, distance_vector < threshold]
        # filtered_pts = cepton_pts.copy()
        # Insert a column of ones
        cepton = np.insert(filtered_pts.T,3,1,axis=1).T
        
        # cepton = transformation_matrix_xy.dot(cepton)
        #cam is an array of {x,y,z}, where x,y are pixels, and z is depth
        # pdb.set_trace()
        # print(transformation_matrix_xy.shape)
        
        column_to_be_added = np.array([[0], [0], [0]])

        # Adding column to array using append() method
        transformation_matrix_xy = np.append(transformation_matrix_xy, column_to_be_added, axis=1)
        transformation_matrix_xy = np.vstack([transformation_matrix_xy, [0, 0, 0, 1]])
        transformation_matrix_cam_xy = Tr_cepton_to_cam.dot(transformation_matrix_xy)
        tm_proj_cam_xy = projection.dot(transformation_matrix_cam_xy)

        cam = tm_proj_cam_xy.dot(cepton)
        # cam = mat_mult(tm_proj_cam_xy, cepton)

        # cam = projection.dot(Tr_cepton_to_cam).dot(cepton)
        # Bassel
        # print("before any deletion:", cepton.shape, cam.shape)
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

        # min_distance = 0
        # max_distance = 45
        # keypoint.distance = zz

        #crops points that are not contained in the image frame
        u_out = np.logical_or(uu<0, uu>IMG_W)
        v_out = np.logical_or(vv<0, vv>IMG_H)
        #list of all points that are not in the frame
        outlier = np.logical_or(u_out, v_out)
        cam = np.delete(cam,np.where(outlier),axis=1)

        cepton = (Tr_cepton_to_cam).dot(cepton)
        # cepton = mat_mult(transformation_matrix_cam_xy, cepton)

        # # pdb.set_trace()
        cepton = np.delete(cepton,np.where(outlier),axis=1)

        xyz = np.zeros((IMG_H+1, IMG_W+1, 3), dtype=np.float16)
        cam = np.round(cam).tolist()
        cam = np.array(cam).astype(np.uint16)
        a = list(zip(cepton[0], cepton[1], cepton[2]))
        # pdb.set_trace()
        length = len(cam[1])
        a = a[:length]
        # print("a: ", len(a),"cam[1]: ", cam[1].shape, "cam[0]: ", cam[0].shape)
        xyz[cam[1], cam[0]] = a
        xyz = xyz[:-1, :-1, :]
        ret[num] = xyz
        # my_global_array[num] = xyz
        # print(np.sum(xyz))
                
        # save data
        label = "{0:.2f}".format((float(self.get_clock().now().nanoseconds)/10**9 - self.globaltime))
        folder = "union-albert"

        # pdb.set_trace()
        # cam_computed = np.sqrt(cepton[0][:]**2+cepton[1][:]**2+cepton[2][:]**2) 
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].scatter([i for i in range(len(cam[2]))], cam[2], c = 'r', s=5)
        # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed[:len(cam[2])], c = 'b', s=5)
        # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed - cam[2], c = 'y', s=5)

        # cv2.imwrite(f"/media/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/dynamic_stop1/images/cam{num}.png", img)

        img = bridge.imgmsg_to_cv2(img, desired_encoding='rgb8')

        # cv2.imwrite(f"/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/{folder}/images/cam{num}/cam{num}_{label}.png", img)
        cv2.imwrite(f"/home/autodrive/workspaces/lid_cam_ws/perceptionData/cam{num}/cam{num}_{label}.png", img)

        with open(f"/home/autodrive/workspaces/lid_cam_ws/perceptionData/pkl{num}/lidar{num}_{label}.pkl", 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(xyz, f, pickle.HIGHEST_PROTOCOL)

        # with open(f"/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/{folder}/pickles/pickle{num}/lidar_{label}.pkl", 'wb') as f:
        # # Pickle the 'data' dictionary using the highest protocol available.
        #     pickle.dump(xyz, f, pickle.HIGHEST_PROTOCOL)
        # pdb.set_trace()
        # # print(time.time()-s)
        # if num == 1000000:
        # print("plotting 2")
        # fig = plt.figure()
        if num ==1:
            plt.axis([0,IMG_W,IMG_H,0])
            plt.imshow(img)
            plt.scatter([uu],[vv],c=[zz],cmap='gist_rainbow',alpha=0.5,s=.5)
            plt.colorbar()
            # plt.show()
            plt.savefig(f'/home/autodrive/Desktop/test/cam'+  str(num) + '.png')
            plt.close()
            # pdb.set_trace()

            name = 'tuning'
            gui = cv2.imread('/home/autodrive/Desktop/test/cam1.png')
            cv2.imshow(name, gui)
            cv2.createTrackbar('tx', name, 0, 1000, on_change_x)
            cv2.createTrackbar('ty', name, 0, 1000, on_change_y)
            cv2.createTrackbar('tz', name, 0, 1000, on_change_z)
            cv2.createTrackbar('theta', name, -7000, 7000, on_change_th)

            cv2.waitKey(0)
            cv2.destroyAllWindows()



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


    def lidar_merge(self, cloudA,cloudB,cloudC):
        thetaAA = 0
        tAAx = 0    #right
        tAAy = 0      #up
        tAAz = 0   # forward
        thetaBA = -np.pi/6
        tBAx = -0.255    #right
        tBAy = 0      #up
        tBAz = -0.061   # forward
        thetaCA = np.pi/6
        tCAx = 0.255    #right
        tCAy = 0      #up
        tCAz = -0.061   # forward

        pointsA = self.merge_one_lidar(cloudA, thetaAA, tAAx, tAAy, tAAz)
        pointsB = self.merge_one_lidar(cloudB, thetaBA, tBAx, tBAy, tBAz)
        pointsC = self.merge_one_lidar(cloudC, thetaCA, tCAx, tCAy, tCAz)
        
        stacked_array12 = np.vstack((pointsA[:,:3],pointsB[:,:3]))
        stacked_array23 = np.vstack((pointsB[:,:3],pointsC[:,:3]))

        return stacked_array12, stacked_array23
        # pdb.set_trace()

        # ax = plt.axes(projection='3d')
        # ax.set(xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
        # ax.scatter(stacked_array[:, 0:1],stacked_array[:, 1:2],stacked_array[:, 2:3],c= 'b',depthshade=False, s=0.01)
        # # ax.scatter(pointsB[:, 0:1],pointsB[:, 1:2],pointsB[:, 2:3],c= 'r',depthshade=False, s=0.01)
        # # ax.scatter(pointsC[:, 0:1],pointsC[:, 1:2],pointsC[:, 2:3],c= 'g',depthshade=False, s=0.01)

        # ax.scatter(0,0,0, s=20)
        # ax.view_init(elev=10,azim=270);
        # plt.xlabel('X [m]')
        # plt.ylabel('Y [m]')
        # # plt.savefig('/home/autodrive/lid_cam_ws/3clouds.png')            
        # plt.pause(0.0001)
        # plt.show()


# def overlay2():
#     global img,num, ret, pc_sensor, pc_centre, slider_tx, slider_ty, slider_tz, slider_theta
#     # P2 (3 x 4) for left eye
#     s = time.time()
#     #get projection matrix
#     # print(pc.shape)
#     R0_rect = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
#     # Add a 1 in bottom-right, reshape to 4 x 4
#     R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
#     R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)

#     # v = [vgnss.x, vgnss.y,vgnss.z]
#     # v = [inspva.north_velocity*np.cos(inspva.azimuth)-inspva.east_velocity*np.sin(inspva.azimuth) , inspva.north_velocity*(-1)*np.sin(inspva.azimuth)-inspva.east_velocity*np.cos(inspva.azimuth), 0]
#     v = [0, 0/3.6, 0]
#     # delta t between camera and lidar closest to it
#     dtcam = (pc_sensor.header.stamp.sec + pc_sensor.header.stamp.nanosec/1e9) - (img.header.stamp.sec + img.header.stamp.nanosec/1e9)
#     # delta t for lidar closest to camera and centre lidar
#     if pc_centre != 0:
#         dtlidar = (pc_centre.header.stamp.sec + pc_centre.header.stamp.nanosec/1e9) - (pc_sensor.header.stamp.sec + pc_sensor.header.stamp.nanosec/1e9)
#     else:
#         dtlidar = 0
#     # dt = 0
#     print("num: ", num, " v: ", v, " dt: ", dtcam)

#     if(num==1):
#         theta = np.pi/6          + slider_theta/1000
#         tx = -0.17 + v[0]*dtcam  + slider_tx/1000   #right
#         ty = -0.0015 + v[2]*dtcam+ slider_ty/1000   #-up
#         tz = 0.009 + v[1]*dtcam  + slider_tz/1000 # forward   
#         projection = np.array([1100.727661, 0.000000, 1028.897396, 0.000000, 1161.720093, 771.434530, 0.000000, 0.000000, 1.000000]).reshape(3,3)

#         theta_i = 0   
#         ti_x = -0.255 + v[0]*dtlidar    #right
#         ti_y = 0 + v[0]*dtlidar      #up
#         ti_z = -0.061 + v[0]*dtlidar   # forward
#         # pcL = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
#         # pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
#         # pc = np.vstack((pcL[:,:3],pcC[:,:3]))

#         #anthony troubleshooting
#         pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
#         # pc = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)

#     elif(num==2):
#         theta = 0+ slider_theta/1000
#         tx = -0.111 + v[0]*dtcam + slider_tx/1000   
#         ty = -0.0015  + v[2]*dtcam + slider_ty/1000   
#         tz = -0.007  + v[1]*dtcam+ slider_tz/1000 
#         projection = np.array([1127.220459, 0.000000, 1030.718004, 0.000000, 1184.828491, 749.131132, 0.000000, 0.000000, 1.000000]).reshape(3,3)

#         theta_i = 0
#         ti_x = 0    #right
#         ti_y = 0      #up
#         ti_z = 0   # forward
#         pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
#     elif(num==3):
#         theta = -np.pi/72+ slider_theta/1000
#         tx = 0.111 + v[0]*dtcam + slider_tx  /1000 
#         ty = -0.0015  + v[2]*dtcam + slider_ty  /1000 
#         tz = -0.007 + v[1]*dtcam+ slider_tz/1000 # 
#         projection = np.array([1131.871948, 0.000000, 1022.029700, 0.000000, 1189.723755, 766.790667, 0.000000, 0.000000, 1.000000]).reshape(3,3)

#         theta_i = 0     
#         ti_x = 0    #right
#         ti_y = 0      #up
#         ti_z = 0   # forward
#         pc = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
#     elif(num==4):
#         theta = -np.pi/6+ slider_theta/1000
#         tx = 0.17 + v[0]*dtcam+ slider_tx /1000  
#         ty = -0.0015 + v[2]*dtcam  + slider_ty  /1000 
#         tz = 0.009 + v[1]*dtcam+ slider_tz/1000 # 
#         projection = np.array([1124.364502, 0.000000, 1004.415648, 0.000000, 1185.257812, 767.772319, 0.000000, 0.000000, 1.000000]).reshape(3,3)

#         theta_i = np.pi/6     
#         ti_x = 0.255    #right
#         ti_y = 0      #up
#         ti_z = -0.061   # forward
#         pcR = self.merge_one_lidar(pc_sensor, theta_i, ti_x, ti_y, ti_z)
#         pcC = self.merge_one_lidar(pc_centre, 0, 0, 0, 0)
#         pc = np.vstack((pcR[:,:3],pcC[:,:3]))


#     Tr_cepton_to_cam = np.array([[np.cos(theta),0,np.sin(theta), tx],
#                                 [0,            1,0,             ty],
#                                 [-np.sin(theta),0,np.cos(theta),tz]])

#     # velo: 0: FWD, 1: LEFT, 2: UP
#     # cepton: 0: RIGHT, 1: FWD, 2: UP
#     # cepton -> velo 

#     cepton_pts = pc.T

#     theta = 0
#     transformation_matrix_y = np.array([[np.cos(theta)  ,0  ,np.sin(theta)],
#                                     [0, 1,0],
#                                     [-np.sin(theta), 0, np.cos(theta)]])

#     # cepton_pts = np.dot(transformation_matrix_y, cepton_pts)

#     theta = + np.pi/2 
#     transformation_matrix_x = np.array([[1            ,0             ,0],
#                                     [0, np.cos(theta),-np.sin(theta)],
#                                     [0, np.sin(theta), np.cos(theta)]])

#     transformation_matrix_xy = np.dot(transformation_matrix_x, transformation_matrix_y)
#     # cepton_pts = transformation_matrix_xy.dot(cepton_pts)

#     # Calculate distances
#     # Calculate the Euclidean distance along the third axis (axis=0)
#     distance_vector = np.linalg.norm(cepton_pts, axis=0)
#     # # Set the threshold distance
#     # # Remove points outside the scoring + 5m radius (40 + 5m)
#     threshold = 45.0

#     # # Filter out the points that exceed the threshold
#     filtered_pts = cepton_pts[:, distance_vector < threshold]
#     # filtered_pts = cepton_pts.copy()
#     # Insert a column of ones
#     cepton = np.insert(filtered_pts.T,3,1,axis=1).T
    
#     # cepton = transformation_matrix_xy.dot(cepton)
#     #cam is an array of {x,y,z}, where x,y are pixels, and z is depth
#     # pdb.set_trace()
#     # print(transformation_matrix_xy.shape)
    
#     column_to_be_added = np.array([[0], [0], [0]])

#     # Adding column to array using append() method
#     transformation_matrix_xy = np.append(transformation_matrix_xy, column_to_be_added, axis=1)
#     transformation_matrix_xy = np.vstack([transformation_matrix_xy, [0, 0, 0, 1]])
#     transformation_matrix_cam_xy = Tr_cepton_to_cam.dot(transformation_matrix_xy)
#     tm_proj_cam_xy = projection.dot(transformation_matrix_cam_xy)

#     cam = tm_proj_cam_xy.dot(cepton)
#     # cam = mat_mult(tm_proj_cam_xy, cepton)

#     # cam = projection.dot(Tr_cepton_to_cam).dot(cepton)
#     # Bassel
#     # print("before any deletion:", cepton.shape, cam.shape)
#     cepton = np.delete(cepton,np.where(cam[2,:]<=0),axis=1)
#     cam = np.delete(cam,np.where(cam[2,:]<=0),axis=1)
#     # # get u,v,z
#     cam[:2] /= cam[2,:]
#     # do projection staff
#     IMG_H = 1536
#     IMG_W = 2048
    
#     # filter point out of canvas
#     uu = cam[0]
#     vv = cam[1]
#     zz = cam[2]

#     # min_distance = 0
#     # max_distance = 45
#     # keypoint.distance = zz

#     #crops points that are not contained in the image frame
#     u_out = np.logical_or(uu<0, uu>IMG_W)
#     v_out = np.logical_or(vv<0, vv>IMG_H)
#     #list of all points that are not in the frame
#     outlier = np.logical_or(u_out, v_out)
#     cam = np.delete(cam,np.where(outlier),axis=1)

#     cepton = (Tr_cepton_to_cam).dot(cepton)
#     # cepton = mat_mult(transformation_matrix_cam_xy, cepton)

#     # # pdb.set_trace()
#     cepton = np.delete(cepton,np.where(outlier),axis=1)

#     xyz = np.zeros((IMG_H+1, IMG_W+1, 3), dtype=np.float16)
#     cam = np.round(cam).tolist()
#     cam = np.array(cam).astype(np.uint16)
#     a = list(zip(cepton[0], cepton[1], cepton[2]))
#     # pdb.set_trace()
#     length = len(cam[1])
#     a = a[:length]
#     # print("a: ", len(a),"cam[1]: ", cam[1].shape, "cam[0]: ", cam[0].shape)
#     xyz[cam[1], cam[0]] = a
#     xyz = xyz[:-1, :-1, :]
#     ret[num] = xyz
#     # my_global_array[num] = xyz
#     # print(np.sum(xyz))
            
#     # save data
#     label = "{0:.2f}".format((float(self.get_clock().now().nanoseconds)/10**9 - self.globaltime))
#     folder = "union-albert"

#     # pdb.set_trace()
#     # cam_computed = np.sqrt(cepton[0][:]**2+cepton[1][:]**2+cepton[2][:]**2) 
#     # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     # axs[0].scatter([i for i in range(len(cam[2]))], cam[2], c = 'r', s=5)
#     # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed[:len(cam[2])], c = 'b', s=5)
#     # axs[0].scatter([i for i in range(len(cam[2]))], cam_computed - cam[2], c = 'y', s=5)

#     # cv2.imwrite(f"/media/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/dynamic_stop1/images/cam{num}.png", img)

#     img = bridge.imgmsg_to_cv2(img, desired_encoding='rgb8')

#     # cv2.imwrite(f"/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/{folder}/images/cam{num}/cam{num}_{label}.png", img)
#     cv2.imwrite(f"/home/autodrive/workspaces/lid_cam_ws/perceptionData/cam{num}/cam{num}_{label}.png", img)

#     with open(f"/home/autodrive/workspaces/lid_cam_ws/perceptionData/pkl{num}/lidar{num}_{label}.pkl", 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#         pickle.dump(xyz, f, pickle.HIGHEST_PROTOCOL)

#     # with open(f"/autodrive/autodrive_2TB_SS/perceptionDataTestTrack/{folder}/pickles/pickle{num}/lidar_{label}.pkl", 'wb') as f:
#     # # Pickle the 'data' dictionary using the highest protocol available.
#     #     pickle.dump(xyz, f, pickle.HIGHEST_PROTOCOL)
#     # pdb.set_trace()
#     # # print(time.time()-s)
#     # if num == 1000000:
#     # print("plotting 2")
#     # fig = plt.figure()
#     if num ==1:
#         plt.axis([0,IMG_W,IMG_H,0])
#         plt.imshow(img)
#         plt.scatter([uu],[vv],c=[zz],cmap='gist_rainbow',alpha=0.5,s=.5)
#         plt.colorbar()
#         # plt.show()
#         plt.savefig(f'/home/autodrive/Desktop/test/cam'+  str(num) + '.png')
#         plt.close()
#         # pdb.set_trace()

#         # name = 'tuning'
#         # gui = cv2.imread('/home/autodrive/Desktop/test/cam1.png')
#         # cv2.imshow(name, gui)
#         # cv2.createTrackbar('tx', name, 0, 1000, on_change_x)
#         # cv2.createTrackbar('ty', name, 0, 1000, on_change_y)
#         # cv2.createTrackbar('tz', name, 0, 1000, on_change_z)
#         # cv2.createTrackbar('theta', name, -7000, 7000, on_change_th)

#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    ros_perception_node = RosPerceptionNode()

    rclpy.spin(ros_perception_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
# transformation_matrix_x