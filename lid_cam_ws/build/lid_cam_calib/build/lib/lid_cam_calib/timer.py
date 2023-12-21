import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32

import time

def timeadjust(time):
    return time*1e9



class MinimalPublisher(Node):

    def delta(self):
        return self.get_clock().now().to_msg().nanosec - self.globaltime.to_msg().nanosec

    def __init__(self):
        super().__init__('minimal_publisher')

        #SM, scoring, 2cb, 1e1, 337, 315, 11
        CAN_group = ReentrantCallbackGroup()
        G1 = MutuallyExclusiveCallbackGroup()
        G2 = MutuallyExclusiveCallbackGroup()
        self.globaltime = self.get_clock().now()

        self.i = 0
        self.j = 0
        self.torque = Float32()
        self.torque_sub = self.create_subscription(Float32, "/torqueinput", self.torque_callback, 10, callback_group=CAN_group)

        self.brake = Float32()
        self.brake_sub = self.create_subscription(Float32, "/brakeinput", self.brake_callback, 10, callback_group=CAN_group)

        self.steering_wheel_angle = Float32()
        self.steering_wheel_angle_sub = self.create_subscription(Float32, "/SWinput", self.steering_wheel_angle_callback, 10, callback_group=CAN_group)
    
        # timer_receive_can = 0.025  # seconds
        # self.receive_can = self.create_timer(timer_receive_can, self.timer_receive_can, callback_group=CAN_group)

        timer_SM = 0.1  # seconds
        self.SM = self.create_timer(timer_SM, self.timer_SM, callback_group=CAN_group)

        timer_scoring = 0.01  # seconds
        self.scoring = self.create_timer(timer_scoring, self.timer_scoring, callback_group=CAN_group)

        # timer_2cb = 0.04  # seconds
        # self._2cb = self.create_timer(timer_2cb, self.timer_2cb, callback_group=CAN_group)

        # timer_1e1 = 0.03  # seconds
        # self._1e1 = self.create_timer(timer_1e1, self.timer_1e1, callback_group=CAN_group)

        # timer_337 = 0.01  # seconds
        # self._337 = self.create_timer(timer_337, self.timer_337, callback_group=CAN_group)

        # timer_315 = 0.04  # seconds
        # self._315 = self.create_timer(timer_315, self.timer_315, callback_group=CAN_group)

        # timer_11 = 0.1  # seconds
        # self._11 = self.create_timer(timer_11, self.timer_11, callback_group=CAN_group)

        #Global Variables for State Machine
        #NOTE CONTROLS CALLBACKS
    def torque_callback(self, msg):
        self.torque = msg

    def brake_callback(self, msg):
        self.brake = msg  
        # self.get_logger().info('Publishing brake: %s ' % self.brake)


    def steering_wheel_angle_callback(self, msg):
        self.steering_wheel_angle = msg
        # self.get_logger().info('Publishing SW: %s ' % self.steering_wheel_angle)
    
        #NOTE RECEIVE CAN CALLBACK
    def timer_receive_can(self):
        self.get_logger().info('')



    def timer_SM(self):
        # rate.sleep(1)
        if ((self.delta() % timeadjust(.1)) < timeadjust(.005)):
        # if ((self.get_clock().now() - self.globaltime)
            self.i+=1
            self.get_logger().info('Publishing SM')



    def timer_scoring(self):
        self.j += 1
        self.get_logger().info('scoring %s' % self.j)

    # def timer_2cb(self): #HS
    #     self.get_logger().info('')
    
    # def timer_1e1(self): #HS
    #     self.get_logger().info('')

    # def timer_337(self): #CE
    #     self.get_logger().info('')

    # def timer_315(self): #CE
    #     self.get_logger().info('')

    # def timer_11(self): #SC
    #     self.get_logger().info('')



def main(args=None):

    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    executor = MultiThreadedExecutor()
    executor.add_node(minimal_publisher)
    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
