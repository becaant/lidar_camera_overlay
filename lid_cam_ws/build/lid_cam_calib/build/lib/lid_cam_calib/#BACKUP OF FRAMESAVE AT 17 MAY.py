#BACKUP OF FRAMESAVE AT 17 MAY

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
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

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

# Custom Message imports
from perception_frame.msg import Frame
from geometry_msgs.msg import Point

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

class FrameNode(Node):

    def __init__(self):
        super().__init__('frame_node')
        # self.subscription = self.create_subscription(Image,'/lucid_vision/camera_1/image',self.im_callback,10)
        self.subscription = self.create_subscription(Image,'/image4',self.im_callback,10)
        self.subscription = self.create_subscription(PointCloud2,'/cepton_pcl2',self.pc_callback,10)

        #IP for lidars is 192.168.x.x netmask 16

        self.subscription  # prevent unused variable warning

        # Publisher setup and execution
        # self.publisher = self.create_publisher(Frame, '/peception_frame', 10)
        # self.timer = self.create_timer(1, self.frame_callback)
        # self.counter = 0

        #declaring data types to be used within the class
        # self.undistorted_img
        # self.x
        # self.y
        # self.z
        # self.intensity
        
        # print(self.undistorted_img)
        # self.overlay()

    def im_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.encoding)
        # Define the camera matrix and distortion coefficients
        camera_matrix = np.array([[1254.926067, 0.000000, 1026.995770], 
                        [0.000000, 1255.840035, 771.593421],
                        [0.000000, 0.000000, 1.000000]])

        distortion_coefficients = np.array([-0.199586, 0.067045, -0.000109, 0.000466, 0.000000])
        width = 2048
        height = 1536

        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (width,height), 0)

        # Undistort the image
        distorted_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.undistorted_img = cv2.undistort(distorted_img, camera_matrix, distortion_coefficients, None, newcameramatrix)

        # Display the undistorted image
        cv2.imwrite("/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/distorted1.png" , distorted_img)
        cv2.imwrite("/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/undistorted1.png" , self.undistorted_img)

    def pc_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.fields[4])
        pcd_np = np.array(list(read_points(msg)))
        # print(pcd_np)

        self.x = pcd_np[:,0:1]
        self.y = pcd_np[:,1:2]
        self.z = pcd_np[:,2:3]
        self.intensity = pcd_np[:,3:4]
        # self.channel = pcd_np[:,7:8]

        print(self.channel)
        # intensity = intensity/max(intensity)
        # cmap=cm.get_cmap('viridis')
        # rgba=cmap(intensity)

        # if (len(x)<len(y)):
        #     if (len(x)<len(z)):
        #         length = len(x)
        #     elif (len(y)<len(z)):
        #         length = len(y)
        # else:
        #     length = len(z)

        # pdb.set_trace()

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set(xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
        # ax.scatter(x[:length],y[:length],z[:length],c=intensity[:length], cmap='viridis',depthshade=False, s=0.01)
        # ax.scatter(0,0,0, s=20)
        # ax.view_init(elev=5,azim=90);
        # plt.xlabel('X [m]')
        # plt.ylabel('Y [m]')
        #plt.show()

        np.savetxt('/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/x.csv', self.x, delimiter = ',')
        np.savetxt('/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/y.csv', self.y, delimiter = ',')
        np.savetxt('/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/z.csv', self.z, delimiter = ',')
        np.savetxt('/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/intensity.csv', self.intensity, delimiter = ',')

    # Publisher Callback
    def frame_callback(self):
        img = Image()
        point = Point()
        msg = Frame()

        img = self.undistorted_img

        pcd_array = []

        for i in range(len(self.length)):
            point.x = self.x_filtered[i]
            point.y = self.y_filtered[i]
            point.z = self.z_filtered[i]

            pcd_array.append(point)

        msg.image = img
        msg.point_array = pcd_array

        self.publisher.publish(msg)


    def overlay(self):

        self.get_logger().info("in overlay")
        x = self.x
        y = self.y
        z = self.z
        intensity = self.intensity
        img = self.undistorted_img

        valid_indices = np.where((x >= -1) & (x <= 1) & (y < 3.5))
        self.x_filtered = x[valid_indices]
        self.y_filtered = y[valid_indices]
        self.z_filtered = z[valid_indices]
        intensity_filtered = intensity[valid_indices]

        # Normalize the intensity values
        scaler = MinMaxScaler()
        intensity_normalized = scaler.fit_transform(intensity_filtered.reshape(-1, 1)).flatten()

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Display the 3D point cloud with normalized intensity values as a color map
        scatter = ax.scatter(self.x_filtered, self.y_filtered, self.z_filtered, c=intensity_normalized, cmap='hot')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Add a colorbar to show the mapping of intensity values to colors
        cbar = plt.colorbar(scatter)

        # Show the plot
        plt.savefig("heat.png")

        x = self.x_filtered
        y = self.y_filtered
        z = self.z_filtered
        intensity = intensity_filtered

        # pdb.set_trace()


        if (len(x)<len(y)):
            if (len(x)<len(z)):
                self.length = len(x)
            elif (len(y)<len(z)):
                self.length = len(y)
        else:
            self.length = len(z)

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

        # #cepton_pts = np.dot(transformation_matrix_z, points.T)
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
        # cepton_pts = cepton_pts.T
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
        count = 0

        for i in intensity:
            if i > 5:
                intensity[count] = 0
            count = count + 1


        colors= preprocessing.normalize([intensity])
        # colors= [intensity]

        plt.scatter([uu],[vv],c=colors,cmap='rainbow_r',s=20)
        # Display the 3D point cloud with intensity values as a color map
        plt.savefig("calib.png")
        #plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
        # plt.show()



def main(args=None):
    rclpy.init(args=args)

    frame_node = FrameNode()

    rclpy.spin(frame_node)
    #frame_node.overlay()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    frame_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
