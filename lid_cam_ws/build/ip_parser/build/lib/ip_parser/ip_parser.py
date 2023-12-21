# STD Imports
import pcapy
import socket

# ROS2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

# STD Varialbles (FOR USE ON THE WORKSTATION USING THE SWITCH)
ethernetPort = "enp3s0"
lidarA_ip = "192.168.73.207"
lidarB_ip = "192.168.74.52"


class IpSubscriber(Node):

    def __init__(self):
        super().__init__('ip_subscriber')
        self.subscription = self.create_subscription(PointCloud2, '/cepton_pcl2', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        if self.isIPactive(ethernetPort, lidarA_ip) == True:
            self.get_logger().info('I heard LiDAR A')
        
        elif self.isIPactive(ethernetPort, lidarB_ip) == True:
            self.get_logger().info('I heard LiDAR B')

        else :
            self.get_logger().info('I didnt hear anything t')

    def isIPactive(self, ethernetPort, ipAddress) -> bool:
        # Open the network interface in promiscuous mode
        cap = pcapy.open_live(ethernetPort, 65536, True, 100)

        # Set a filter to capture packets from the specific IP address
        filter_str = f"src host {ipAddress}"
        cap.setfilter(filter_str)

        # Capture a single packet
        _, packet = cap.next()

        
        if packet is not None:
            return True
        else:
            return False
        

def main(args=None):
    rclpy.init(args=args)
    ip_subscriber = IpSubscriber()
    rclpy.spin(ip_subscriber)
    ip_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()