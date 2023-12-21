#!/usr/bin/env python 3.8.10
import rclpy
from rclpy.node import Node
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber

import math
import numpy as np
import matplotlib.pyplot as plt
import struct
from mpl_toolkits import mplot3d
import time
import open3d as o3d

from sensor_msgs.msg import PointCloud2, PointField

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

class Pc_Sub(Node):

    def __init__(self):
        super().__init__('pc_sub')
        self.subA = Subscriber(self, PointCloud2, "cepton_pcl2")
        # self.subB = Subscriber(self, PointCloud2, "cepton_pcl2B")
        # self.subC = Subscriber(self, PointCloud2, "cepton_pcl2C")

        # self.ts = ApproximateTimeSynchronizer([self.subA, self.subB, self.subC],10, 0.05)
        self.ts = ApproximateTimeSynchronizer([self.subA],10, 0.05)
        self.ts.registerCallback(self.pc_callback)

    def pc_callback(self, cloudA):
    # def pc_callback(self, cloudA, cloudB, cloudC):
            self.get_logger().info("Hello")
            # start_time = time.time
            pcd_npA = np.array(list(read_points(cloudA)))
            # pcd_npB = np.array(list(read_points(cloudB)))
            # pcd_npC = np.array(list(read_points(cloudC)))
            # print(pcd_np)

            xA = pcd_npA[:,0:1]
            yA = pcd_npA[:,1:2]
            zA = pcd_npA[:,2:3]
            iA = pcd_npA[:,3:4]
            pointsA = np.column_stack((xA, yA, zA))
 
            import pdb;pdb.set_trace()
            o3d.io.write_point_cloud("autodrive_point_cloud.pcd", pointsA)

            # xB = pcd_npB[:,0:1]
            # yB = pcd_npB[:,1:2]
            # zB = pcd_npB[:,2:3]
            # pointsB = np.column_stack((xB, yB, zB))

            # xC = pcd_npC[:,0:1]
            # yC = pcd_npC[:,1:2]
            # zC = pcd_npC[:,2:3]
            # pointsC = np.column_stack((xC, yC, zC))

            # intensity = pcd_np[:,3:4]

            # if (len(x)<len(y)):
            #     if (len(x)<len(z)):
            #         length = len(x)
            #     elif (len(y)<len(z)):
            #         length = len(y)
            # else:
            #     length = len(z)
            print('aaa')
            # fig = plt.figure(1)
            # ax = plt.axes(projection='3d')
            # ax.set(xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
            # ax.scatter(xA,yA,zA,c= 'b',depthshade=False, s=0.01)
            # # ax.scatter(xB,yB,zB,c= 'r',depthshade=False, s=0.01)
            # # ax.scatter(xC,yC,zC,c= 'g',depthshade=False, s=0.01)
            # # ax.scatter(pointsA[0],pointsA[1],pointsA[2],c= 'b',depthshade=False, s=0.01)
            # # ax.scatter(pointsB[0],pointsB[1],pointsB[2],c= 'r',depthshade=False, s=0.01)
            # # ax.scatter(pointsC[0],pointsC[1],pointsC[2],c= 'g',depthshade=False, s=0.01)
            # ax.scatter(0,0,0, s=20)
            # ax.view_init(elev=5,azim=90);
            # plt.xlabel('X [m]')
            # plt.ylabel('Y [m]')
            # # plt.savefig('/home/autodrive/lid_cam_ws/src/lid_cam_calib/lid_cam_calib/3clouds.png')            
            # plt.pause(0.0001)
            # plt.show()



def main(args=None):
    rclpy.init(args=args)
    pc_sub = Pc_Sub()

    rclpy.spin(pc_sub)
    rclpy.shutdown()


if __name__ == '__main__':
    main()