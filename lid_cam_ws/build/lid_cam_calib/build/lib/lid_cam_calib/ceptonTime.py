import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subpc = self.create_subscription(PointCloud2,'cepton_pcl2',self.pc_callback,10)
        self.subim = self.create_subscription(Image,'lucid_vision/camera_1/image',self.im_callback,10)

        self.newheader = self.get_clock().now().to_msg()
        print(type(self.newheader))

        self.pub = self.create_publisher(PointCloud2, 'ceptime', 10)

    def im_callback(self, msg):
        self.newheader = msg.header.stamp
        # self.pub.publish(msg)
    
    def pc_callback(self,msg):
        msg.header.stamp = self.newheader
        self.get_logger().info('publishing pc')
        self.pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()