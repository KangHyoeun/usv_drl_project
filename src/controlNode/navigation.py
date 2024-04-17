#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile

import transformations
from geometry_msgs.msg import Quaternion, Point, Twist, Vector3, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from .tool.navigtion_tool import KalmanFilter, LowPassFilter

class Navigation(Node):
    def __init__(self):
        super().__init__('navigation_node')
        qos_profile = QoSProfile(depth=10)
        
        # Initial data
        self.dt                = 0.1
        self.slam_msg          = None
        self.imu_msg           = None
        self.prev_slam_time    = None
        self.last_slam_time    = None
        self.prev_system_time  = None

        # Subscriber
        self.slam_subscriber = self.create_subscription(Odometry, '/odom', self.slam_callback, qos_profile)
        self.slam_subscriber2 = self.create_subscription(PoseStamped, '/robot1/current_pose', self.slam2_callback, qos_profile)
        self.imu_subscriber = self.create_subscription(Imu, '/nav/filtered_imu/data', self.imu_callback, qos_profile)

        # Publisher       
        self.navigation_LPF_publisher = self.create_publisher(Odometry, 'navigation/LPF', qos_profile)
        self.navigation_EKF_publisher = self.create_publisher(Odometry, 'navigation/EKF', qos_profile)
        
        # Get Parameter
        self.declare_parameter('filter', None)
        self.get_param()

        # Set timer
        self.timer = self.create_timer(self.dt, self.publish_navigation_msg)
        

    def get_param(self):
        self.filter      = self.get_parameter('filter').value

    def slam_callback(self, msg):
        self.slam2_msg = msg

    def slam2_callback(self, msg):
        self.slam_msg = msg

    def imu_callback(self, msg):
        self.imu_msg = msg
                        
    def publish_navigation_msg(self):
        if self.imu_msg is None or self.slam_msg is None:
            return
        # if self.slam_msg is None:
        #     return
        #* 현재 시간
        time_now = self.get_clock().now()
        system_time = time_now.seconds_nanoseconds()[0] + time_now.seconds_nanoseconds()[1] / 1e9

        #* state
        x = self.slam_msg.pose.position.x
        y = -self.slam_msg.pose.position.y
        
        orientation = self.slam_msg.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        psi, pitch, roll = transformations.euler_from_quaternion(quaternion)
        psi =-psi
        # self.get_logger().info(f'psi {round(roll*180.0/np.pi,1)}, psi2 {round(psi*180.0/np.pi,1)}')
        # u = self.slam_msg.twist.twist.linear.x
        # v = self.slam_msg.twist.twist.linear.y
        # r = self.slam_msg.twist.twist.angular.z

        # 각속도
        angular_vel_x = self.imu_msg.angular_velocity.x
        angular_vel_y = self.imu_msg.angular_velocity.y
        angular_vel_z = self.imu_msg.angular_velocity.z

        # 각속도 공분산
        angular_vel_cov = self.imu_msg.angular_velocity_covariance

        # 선속도 
        linear_accel_x = self.imu_msg.linear_acceleration.x
        linear_accel_y = self.imu_msg.linear_acceleration.y
        linear_accel_z = self.imu_msg.linear_acceleration.z

        # 선속도 공분산 
        linear_accel_cov = self.imu_msg.linear_acceleration_covariance

        #* Filter 적용
        slam_time = self.slam_msg.header.stamp.sec + self.slam_msg.header.stamp.nanosec / 1e9

        if self.prev_slam_time is None:
            # LPF 초기화
            self.LPF = LowPassFilter()
            vel_LPF = self.LPF.update_velocity(np.array([x, y, psi]),angular_vel_z, 0.1, weight = 0.7)

            # EKF 초기화
            self.EKF = KalmanFilter()
            self.EKF.x = np.matrix([x, y, 0, 0, 0, 0, roll, pitch, psi]).reshape(9,1)
            # 데이터 시간 저장
            self.prev_slam_time = slam_time
            self.last_slam_time = system_time

            # 시스템 시간 저장 
            self.prev_system_time = system_time

            # reset parameter
            self.imu_msg = None
            return
        
        # SLAM Data 처리
        if self.prev_slam_time == slam_time: # slam 데이터의 공백
            slam_time_gap = (system_time - self.last_slam_time)
            # SLAM topic 2초간 못 받을 시 오류 출력
            if slam_time_gap > 1.0:
                self.get_logger().info(f'SLAM NO DATA:{round(slam_time_gap,2)} sec')
            # 오차 공분산 설정
            R_w = 0.01 # 시간 가중치
            R = np.matrix( 0.01* np.eye(9, 9))
            
        else: # slam 데이터를 수신
            #LPF
            vel_LPF = self.LPF.update_velocity(np.array([x, y, psi]),angular_vel_z, 0.1, weight = 0.7)

            odom_LPF = Odometry()
            # odom_LPF.header.stamp = time_now.to_msg()

            position = Point(x=float(x), y=float(y), z=float(0))
            odom_LPF.pose.pose.position = position

            linear_velocity = Vector3(x=float(vel_LPF[0]), y=float(vel_LPF[1]), z=float(0))
            odom_LPF.twist.twist.linear = linear_velocity
            # self.get_logger().info(f'u {round(vel_LPF[0],1)}, v {round(vel_LPF[1],1)}, psi {round(psi*180.0/np.pi,1)}, r {round(vel_LPF[2],1)}')
            angular_velocity = Vector3(x=float(0), y=float(0), z=float(vel_LPF[2]))
            odom_LPF.twist.twist.angular = angular_velocity
            odom_LPF.pose.pose.orientation = orientation

            self.navigation_LPF_publisher.publish(odom_LPF)
            # EKF
            R = np.matrix(0.01 * np.eye(9, 9))
            # 데이터 시간 저장
            self.prev_slam_time = slam_time
            self.last_slam_time = system_time

        # SLAM 측정값
        z = np.matrix([x, y, 0,float(self.LPF.vel[0]),float(self.LPF.vel[1]),0, roll, pitch, psi]).reshape(9,1)
        # IMU Data 처리 
        u = np.matrix([linear_accel_x, linear_accel_y, 0, angular_vel_x, angular_vel_y, angular_vel_z]).reshape(6,1)
        # 오차 공분산 설정
        Q = np.matrix(np.eye(6, 6))
        Q[0:3, 0:3] = linear_accel_cov.reshape((3,3))
        Q[3:6, 3:6] = angular_vel_cov.reshape((3,3))
        # EKF 상태 업데이트
        self.EKF.dt = system_time - self.prev_system_time

        self.EKF.update(u, Q, z, R)
        state_ekf = self.EKF.x
        self.prev_system_time = system_time

        # publish
        odom_EKF = Odometry()
        # odom_EKF.header.stamp = time_now.to_msg()

        position = Point(x=float(state_ekf[0]), y=float(state_ekf[1]), z=float(state_ekf[2]))
        odom_EKF.pose.pose.position = position

        linear_velocity = Vector3(x=float(state_ekf[3]), y=float(state_ekf[4]), z=float(state_ekf[5]))
        odom_EKF.twist.twist.linear = linear_velocity

        euler = state_ekf[6:9]
        q = transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
        odom_EKF.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.navigation_EKF_publisher.publish(odom_EKF)

        # reset parameter
        self.imu_msg = None

        
def main(args=None):
    rclpy.init(args=args)
    try:
        navigation_node = Navigation()
        try:
            rclpy.spin(navigation_node)
        except KeyboardInterrupt:
            navigation_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            navigation_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
