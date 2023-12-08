#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import kinematics as kin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cv2
from ament_index_python.packages import get_package_share_directory

"""
Code to trace out the contours of an image
OpenCV is used to get the contours of the image in the /images foldes
The contours are returned as a numpy array, which the end-effector traces out
ros2 run painter toolpath_controller.py
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

        ##### Define a line trajectory #####
        
        self.num_points = 500
        
        ##### #####
        # cv2_imshow(img)
        
        img_path = get_package_share_directory('painter') + '/images/' + 'amongus.png'
        image = cv2.imread(img_path)
        
        # image = cv2.imread('images/amongus.png')
        
        print(image)
        # Convert image to gray scale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        contours, heirarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        scaling = 500

        contour1 = contours[1]
        x_1 = contour1[:, 0][:, 0] / scaling
        y_1 = contour1[:, 0][:, 1] / scaling
        z_1 = np.zeros(len(y_1))

        contour2 = contours[2]
        x_2 = contour2[:, 0][:, 0] / scaling
        y_2 = contour2[:, 0][:, 1] / scaling
        z_2 = np.zeros(len(y_2))

        x_d = np.concatenate((x_1, x_2)) 
        y_d = np.concatenate((y_1, y_2))
        z_d = np.concatenate((z_1, z_2))

        # Minor reorientation
        x_d = x_d - x_d[0]
        epsilon = -1e-5

        ref_traj = np.column_stack((x_d, y_d, z_d, np.ones(len(z_d))))

        transform = kin.T0_n_lambdify(self.q[0, 0], self.q[1, 0], self.q[2, 0],
                                      self.q[3, 0], self.q[4, 0], self.q[5, 0])
        # print(np.around(transform))
        ref_traj = transform @ ref_traj.T

        self.x_des = ref_traj[0, :]
        self.y_des = ref_traj[1, :]
        self.z_des = ref_traj[2, :] 

        # Go from current configuration to start point of painting
        x_ = np.linspace(transform[0, 3], self.x_des[0], 100)
        y_ = np.linspace(transform[1, 3], self.y_des[0], 100)
        z_ = np.linspace(transform[2, 3], self.z_des[0], 100)

        self.x_des = np.concatenate((x_, self.x_des))
        self.y_des = np.concatenate((y_, self.y_des))
        self.z_des = np.concatenate((z_, self.z_des))
        
        self.dt = 20/self.num_points
        self.i = 0

        ##### See the plot once before trajectory execution #####
        # fig = plt.figure()
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
        # print(transform[0, 3], transform[1, 3], transform[2, 3])
        # # Adjust the marker size (s) to reduce the circle size
        # ax.scatter(self.x_des, self.y_des, self.z_des, c='black', label='Reference Trajectory', linewidth=0.01, s=3)
        # ax.scatter(transform[0, 3], transform[1, 3], transform[2, 3], c='blue', label='Starting Point', marker='x', linewidth=5, s=3)
        # ax.set_ylim(0, 1)
        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')
        # ax.set_title('3D plot of trajectory')
        # # Show the plot
        # plt.legend()
        # plt.show()


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

            Ve = np.array([((self.x_des[self.i+1] - x)/self.dt, 
                            (self.y_des[self.i+1] - y)/self.dt,
                            (self.z_des[self.i+1] - z)/self.dt, 0, 0, 0)]).astype(np.float64).T
            
            print("Ve: ", np.around(Ve.T, 3))

            # q_dot = J_inv @ Ve
            J = np.array(kin.j(q_).tolist()).astype(np.float64)
            # print(J)
            q_dot = J.T @ np.linalg.inv(J @ J.T + self.lambda_**2 * np.identity(6)) @ Ve
            # q_dot = J.T @ Ve
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

        else:
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')

            # Adjust the marker size (s) to reduce the circle size
            ax.scatter(x_plot, y_plot, z_plot, c='red', label='End Effector Trajectory', linewidth=1, s=3)
            ax.scatter(x_des_plot, y_des_plot, z_des_plot, c='black', label='Reference Trajectory', linewidth=1, s=3)
            ax.set_ylim(0, 1)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('3D plot of trajectory')

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


