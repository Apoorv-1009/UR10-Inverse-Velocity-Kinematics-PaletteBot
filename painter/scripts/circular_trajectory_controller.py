#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import kinematics as kin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Code to draw a circle of radius 10cm
Circle is drawn at the origin, which we Transform in terms of the end-effector coordinates
by multiplying it with the Final Transformation Matrix
ros2 run painter circular_trajectory_controller.py
"""

x_plot, y_plot, z_plot = [], [], []
x_des_plot, y_des_plot, z_des_plot = [], [], []

class PositionControllerNode(Node):

    def __init__(self):
        epsilon = -1e-5
        self.q = np.array([epsilon, epsilon, epsilon, epsilon, epsilon, epsilon], dtype=np.float64).reshape((6, 1))
        
        self.lambda_ = 0.05

        super().__init__("proportional_controller")
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10)

        ##### Define a circular trajectory #####
        self.num_points = 500
        radius = 0.1
        angle = np.linspace(0, 2*np.pi, self.num_points)

        self.x_des = radius * np.sin(angle) 
        self.y_des = radius * -np.cos(angle) + radius
        self.z_des = np.zeros(len(self.y_des)) 

        ref_traj = np.column_stack((self.x_des, self.y_des, self.z_des, np.ones(len(self.z_des))))
        transform = kin.T0_n_lambdify(self.q[0, 0], self.q[1, 0], self.q[2, 0],
                                      self.q[3, 0], self.q[4, 0], self.q[5, 0])
        
        ref_traj = transform @ ref_traj.T

        self.x_des = ref_traj[0, :]
        self.y_des = ref_traj[1, :]
        self.z_des = ref_traj[2, :]
        print(self.y_des[100])
        
        self.dt = 20/self.num_points
        self.i = 0


        ##### Create Subscriber #####
        
         # Joint State subscriber
        self.joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, qos_profile)
                

        ##### Create publishers #####

        # Joint angle publisher
        self.position_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 10)
        
        ##### Create timers to call position_publisher #####
        
        self.timer = self.create_timer(self.dt, self.position_publisher)

    def position_publisher(self):
        q_ = self.q

        if self.i < len(self.x_des) -2:
        # if self.i < 10:

            print("i: ", self.i)
            # print("q: ", np.around(q_.T, 3))
            transform = np.array(kin.fk_solver(q_).tolist()).astype(np.float64)
            
            # print("transform: \n", transform)

            x, y, z = np.around(transform[0, 3], 3), np.around(transform[1, 3], 3), np.around(transform[2, 3], 3)
            x_plot.append(x), y_plot.append(y), z_plot.append(z)
            x_des_plot.append(self.x_des[self.i]), y_des_plot.append(self.y_des[self.i]), z_des_plot.append(self.z_des[self.i]) 
            print("x, y, z: ", x, y, z)
            print("x_des, y_des, z_des: ", np.around(self.x_des[self.i+1], 3), 
                                           np.around(self.y_des[self.i+1], 3),
                                           np.around(self.z_des[self.i+1], 3))

            # J_inv = np.array(kin.j_inv(q_).tolist()).astype(np.float64)

            Ve = np.array([((self.x_des[self.i+1] - x)/self.dt, 
                            (self.y_des[self.i+1] - y)/self.dt,
                            (self.z_des[self.i+1] - z)/self.dt, 0, 0, 0)]).astype(np.float64).T

            print("Ve: ", np.around(Ve.T, 3))

            # q_dot = J_inv @ Ve
            J = np.array(kin.j(q_).tolist()).astype(np.float64)
            # print(J)
            q_dot = J.T @ np.linalg.inv(J @ J.T + self.lambda_**2 * np.identity(6)) @ Ve
            # q_dot = J.T @ Ve
            # print(type(q_dot))
            print("q_dot: ", np.around(q_dot.T, 3))

            q_ += q_dot * self.dt
            print("q_: ", np.around(q_.T, 3))

            angle = Float64MultiArray()
            angle.data = [q_[0, 0], q_[1, 0], q_[2, 0], q_[3, 0], q_[4, 0], q_[5, 0]]
            self.position_pub.publish(angle)

            self.i += 1

            print("error: ", self.x_des[self.i]-x,
                             self.y_des[self.i]-y,
                             self.z_des[self.i]-z)
            if(self.x_des[self.i]-x > 1 or self.y_des[self.i]-y > 1 or self.z_des[self.i]-z > 1):
                print("ERROR")
                print("#################################################################################################################################################################################################################################################")
                self.i = len(self.x_des)
                
            
            print("\n")
            # self.q = q_

        else:
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

            # Adjust the marker size (s) to reduce the circle size
            ax.scatter(x_plot, y_plot, z_plot, c='red', label='End Effector Trajectory', linewidth=1, s=3)
            ax.scatter(x_des_plot, y_des_plot, z_des_plot, c='black', label='Reference Trajectory', linewidth=1, s=3)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('3D Plot of Circle')
            ax.set_ylim(0, 1)

            # Show the plot
            plt.legend()
            plt.show()

    def joint_state_cb(self, msg):
        # The message format may seem weird, but upon 
        # doing ros2 topic echo, this format is revealed
        self.q = np.array([msg.position[2], msg.position[0], msg.position[1], 
                           msg.position[3], msg.position[4], msg.position[5]], dtype=np.float64).reshape((6, 1))
        
        # transform = np.array(kin.fk_solver(self.q).tolist()).astype(np.float64)
        # print("x, y, z :", np.around(transform[0, 3], 3), np.around(transform[1, 3], 3), np.around(transform[2, 3], 3))
        # print("error: ", np.around(self.x_des[self.i]-transform[0, 3], 3),
        #                  np.around(self.y_des[self.i]-transform[1, 3], 3),
        #                  np.around(self.z_des[self.i]-transform[2, 3], 3))
        # print("Transform: \n", transform)

        

def main(args=None):
    rclpy.init(args=args)
    node = PositionControllerNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


