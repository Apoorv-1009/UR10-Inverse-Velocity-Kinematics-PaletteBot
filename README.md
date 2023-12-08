# UR10-Inverse-Velocity-Kinematics-PaletteBot

This ROS2 package provides control scripts for a robotic arm equipped with a customized gripper, designed for automated painting tasks. The package utilizes inverse velocity kinematics for precise control of the end-effector. For producing a stable jacobian around singularity positions, the Damped Least Squares Inverse method is used

## Usage
Launch the gazebo world using: 
- `ros2 launch painter gazebo.launch.py`

To run the different controllers, use the following commands:
- `ros2 run painter wall_painter_controller.py`: Paint along a wall using a linear trajectory.
- `ros2 run painter toolpath_controller.py`: Trace contours of an image using OpenCV.
- `ros2 run painter circular_trajectory_controller.py`: Draw a circle of radius 10cm at the origin.
- `ros2 run painter line_trajectory_controller.py`: Track a linear path from the initial position to a desired point.

## Controller Descriptions

### wall_painter_controller

This controller implements a linear trajectory along a wall. It plans a linear path between the end-effector position and the starting point of the wall. The `add_segment()` function is used to plan a wall trajectory between a set of points.

### toolpath_controller

This controller traces contours of an image using OpenCV. The contours are obtained from images in the `/images` folder and returned as a numpy array. The end-effector then traces out these contours.

### circular_trajectory_controller

This controller draws a circle of radius 10cm. The circle is initially drawn at the origin and then transformed in terms of the end-effector coordinates by multiplying it with the Final Transformation Matrix.

### line_trajectory_controller

This controller tracks a linear path from the initial position of the end-effector to a desired point.

## Demonstration
![apoorv sussy](https://github.com/Apoorv-1009/UR10-Inverse-Velocity-Kinematics-PaletteBot/assets/57452076/02087fea-bf41-4b57-b08e-9c25b13e7740)


