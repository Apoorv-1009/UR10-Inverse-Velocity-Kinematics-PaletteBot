"""
File to compute Forward Kinematics and Inverse Velocity Kinematics Components for given
DH parameters of a 6-DOF robotic arm
"""
from sympy import *
import numpy as np

# Define the DH parameter symbols
d, a, alpha, t = symbols('d, a, alpha, t')
theta, theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta, theta1, theta2, theta3, theta4, theta5, theta6', cls=Function)
theta, theta1, theta2, theta3, theta4, theta5, theta6 = theta(t), theta1(t), theta2(t), theta3(t), theta4(t), theta5(t), theta6(t)
g = 9.81

def deg2rad(angle):
  radians = angle * pi / 180
  return radians

# Define the DH Transformation matrix
A = Matrix([
    [cos(theta), -sin(theta) * cos(alpha),   sin(theta) * sin(alpha),    a * cos(theta)],
    [sin(theta), cos(theta) * cos(alpha),    -cos(theta) * sin(alpha),   a * sin(theta)],
    [0,          sin(alpha),                 cos(alpha),                 d],
    [0,          0,                          0,                          1]
])

# Define your DH Parameters, here we will define parameters for the UR10
link1_dh_params = {theta: theta1,                 d: 0.128,    a: 0,        alpha: deg2rad(-90)}
link2_dh_params = {theta: theta2 - deg2rad(90),   d: 0.176,    a: 0.6127,   alpha: deg2rad(180)}
link3_dh_params = {theta: theta3,                 d: 0.1639,   a: 0.5716,   alpha: deg2rad(180)}
link4_dh_params = {theta: theta4 + deg2rad(90),   d: 0.1639,   a: 0,        alpha: deg2rad(90)}
link5_dh_params = {theta: theta5,                 d: 0.1157,   a: 0,        alpha: deg2rad(-90)}
linkn_dh_params = {theta: theta6,                 d: 0.268,    a: 0,        alpha: deg2rad(0)}

# Write the transformation between each link
A0_1 = N(A.subs(link1_dh_params))
A1_2 = N(A.subs(link2_dh_params))
A2_3 = N(A.subs(link3_dh_params))
A3_4 = N(A.subs(link4_dh_params))
A4_5 = N(A.subs(link5_dh_params))
A5_n = N(A.subs(linkn_dh_params))

# Calculate Transform between Base frame and every other frame
T0_1 = N(A0_1) 
T0_2 = N(A0_1 * A1_2)
T0_3 = N(A0_1 * A1_2 * A2_3)
T0_4 = N(A0_1 * A1_2 * A2_3 * A3_4)
T0_5 = N(A0_1 * A1_2 * A2_3 * A3_4 * A4_5)
T0_n = N(A0_1 * A1_2 * A2_3 * A3_4 * A4_5 * A5_n)

# Calculate the Jacobian
Z0_1 = T0_1[:-1, 2]
Z0_2 = T0_2[:-1, 2]
Z0_3 = T0_3[:-1, 2]
Z0_4 = T0_4[:-1, 2]
Z0_5 = T0_5[:-1, 2]
Z0_n = T0_n[:-1, 2]

P0_n = T0_n[:-1, 3]

dp_dtheta1 = diff(P0_n, theta1)
dp_dtheta2 = diff(P0_n, theta2)
dp_dtheta3 = diff(P0_n, theta3)
dp_dtheta4 = diff(P0_n, theta4)
dp_dtheta5 = diff(P0_n, theta5)
dp_dtheta6 = diff(P0_n, theta6)

J_v = dp_dtheta1.row_join(dp_dtheta2)
J_v = J_v.row_join(dp_dtheta3)
J_v = J_v.row_join(dp_dtheta4)
J_v = J_v.row_join(dp_dtheta5)
J_v = J_v.row_join(dp_dtheta6)

J_w = Z0_1.row_join(Z0_2)
J_w = J_w.row_join(Z0_3)
J_w = J_w.row_join(Z0_4)
J_w = J_w.row_join(Z0_5)
J_w = J_w.row_join(Z0_n)

J = J_v.col_join(J_w)

# Use lambda functions to call our kinematics equations
J_lambdify = lambdify([theta1, theta2, theta3, theta4, theta5, theta6], J)
T0_n_lambdify = lambdify([theta1, theta2, theta3, theta4, theta5, theta6], T0_n)

def fk_solver(q):
    """
        Function to compute the Forward Kinematics at a given joint state
        Inputs: joint angles q = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64).reshape((6, 1)) 
        Outputs: End Effector Transform w.r.t base link
    """
    return T0_n_lambdify(q[0, 0], q[1, 0], q[2, 0], q[3, 0], q[4, 0], q[5, 0])

def j(q):
   """
        Function to compute the Jacobian at a given joint state
        Inputs: joint angles q = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64).reshape((6, 1)) 
        Outputs: Jacobian
   """
   return J_lambdify(q[0, 0], q[1, 0], q[2, 0], q[3, 0], q[4, 0], q[5, 0])

def j_inv(q):
    """
        Function to compute inverse jacobian at a given joint state
        Inputs: joint angles q = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64).reshape((6, 1)) 
        Outputs: Inverse Jacobian
    """
    return np.linalg.pinv(J_lambdify(q[0, 0], q[1, 0], q[2, 0], q[3, 0], q[4, 0], q[5, 0]))
    # return   N(J.subs({theta1: q[0, 0], theta2: q[1, 0], theta3: q[2, 0],
    #                    theta4: q[3, 0], theta5: q[4, 0], theta6: q[5, 0]}).pinv())
    
  
  



