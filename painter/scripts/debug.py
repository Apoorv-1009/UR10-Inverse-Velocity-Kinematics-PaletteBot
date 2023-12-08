#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math

"""
Code to publish joint angles
Useful for debugging, setting the UR10 back to home position 
or any random orientation we want
"""

class fk_publish_node(Node):
    def __init__(self):
        super().__init__('fk_publish_node')
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.fk_pub)
        # self.theta1 = math.radians(3)
        # self.theta2 = math.radians(30)
        # self.theta3 = math.radians(8)
        # self.theta4 = math.radians(12)
        # self.theta5 = math.radians(43)
        # self.theta6 = math.radians(0)
        self.theta1 = math.radians(0)
        self.theta2 = math.radians(0)
        self.theta3 = math.radians(0)
        self.theta4 = math.radians(0)
        self.theta5 = math.radians(0)
        self.theta6 = math.radians(0)

    def fk_pub(self):
        joint_positions = Float64MultiArray()
        joint_positions.data = [self.theta1,self.theta2,self.theta3,self.theta4,self.theta5,self.theta6]
        self.joint_position_pub.publish(joint_positions)

def main(args=None):
    rclpy.init(args=args)
    node = fk_publish_node()
    node.fk_pub()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()