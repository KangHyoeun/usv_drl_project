#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile

import transformations

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Header

from .tool.control_tool import *
from .tool.navigtion_tool import *

class Control(Node):
    def __init__(self):
        super().__init__('control_test_node')
        qos_profile = QoSProfile(depth=10)
        
        # Initial subscriber data
        self.navigation_msg = None
        
        # Initial publisher data
        self.dt                = 0.1

        # Subscriber
        self.naviagation_subscriber = self.create_subscription(Odometry, '/navigation/LPF', self.navigation_callback, qos_profile)

        # Publisher       
        self.control_publisher = self.create_publisher(Float64MultiArray, '/control', qos_profile)
        
        # Get Parameter
        self.declare_parameter('mode', None)
        self.declare_parameter('pwm', None)
        self.declare_parameter('rpm', None)
        self.declare_parameter('pwm_limit', None)
        self.declare_parameter('rpm_limit', None)
        self.declare_parameter('rudder_limit', None)
        self.declare_parameter('rudder', None)
        self.declare_parameter('desire_u', None)
        self.declare_parameter('desire_v', None)
        self.declare_parameter('desire_r', None)
        self.declare_parameter('desire_psi', None)
        self.declare_parameter('gain_u', None)
        self.declare_parameter('gain_v', None)
        self.declare_parameter('gain_r', None)
        self.declare_parameter('gain_psi', None)

        self.get_param()

        # init controller
        self.PID_u = PIDControl()
        self.PID_v = PIDControl()
        self.PID_r = PIDControl()
        self.PID_psi = PIDControl()

        self.rpm_port = 37.0
        self.rpm_stbd = -125.0
        self.prev_dt = None

        # Set timer
        self.timer = self.create_timer(0.1, self.publish_control_msg)
    

    def get_param(self):
        self.mode           = self.get_parameter('mode').value
        self.pwm            = self.get_parameter('pwm').value
        self.rpm            = self.get_parameter('rpm').value
        self.rudder         = self.get_parameter('rudder').value
        self.pwm_limit      = self.get_parameter('pwm_limit').value
        self.rpm_limit      = self.get_parameter('rpm_limit').value
        self.rudder_limit   = self.get_parameter('rudder_limit').value
        self.desire_u       = self.get_parameter('desire_u').value
        self.desire_v       = self.get_parameter('desire_v').value
        self.desire_r       = self.get_parameter('desire_r').value
        self.desire_psi       = self.get_parameter('desire_psi').value
        
        self.gain_u         = self.get_parameter('gain_u').value
        self.gain_v         = self.get_parameter('gain_v').value
        self.gain_r         = self.get_parameter('gain_r').value
        self.gain_psi         = self.get_parameter('gain_psi').value


    def navigation_callback(self, msg):
        self.navigation_msg = msg
        
    def publish_control_msg(self):
        if self.navigation_msg is None:
            return
    
        # self.get_logger().info(f'Test Mode {self.mode}')

        # header update
        self.header = Header()
        self.header.stamp = self.get_clock().now().to_msg()


        #* 현재 시간
        time_now = self.get_clock().now()
        system_time = time_now.seconds_nanoseconds()[0] + time_now.seconds_nanoseconds()[1] / 1e9
        if self.prev_dt == None:
            self.prev_dt = system_time


        self.dt = system_time-self.prev_dt

        # Get parameter
        self.get_param()

        # state
        x = self.navigation_msg.pose.pose.position.x
        y = self.navigation_msg.pose.pose.position.y

        orientation = self.navigation_msg.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, psi = transformations.euler_from_quaternion(quaternion)

        u = self.navigation_msg.twist.twist.linear.x
        v = self.navigation_msg.twist.twist.linear.y
        r = self.navigation_msg.twist.twist.angular.z

        # mode
        if self.mode == 1:
            pwm         = saturation(self.pwm, self.pwm_limit[0], self.pwm_limit[1])
            rpm_port    = saturation(self.rpm[0], self.rpm_limit[0], self.rpm_limit[1])
            rpm_stbd    = saturation(self.rpm[1], self.rpm_limit[0], self.rpm_limit[1])
            rudder_port = saturation(self.rudder[0], self.rudder_limit[0], self.rudder_limit[1])
            rudder_stbd = saturation(self.rudder[1], self.rudder_limit[0], self.rudder_limit[1])
                
        elif self.mode == 2:
            # PID gain update
            self.PID_u.update(self.gain_u[0], self.gain_u[1], self.gain_u[2])
            self.PID_v.update(self.gain_v[0], self.gain_v[1], self.gain_v[2])
            self.PID_r.update(self.gain_r[0], self.gain_r[1], self.gain_r[2])
            self.PID_psi.update(self.gain_psi[0], self.gain_psi[1], self.gain_psi[2])
            # error 
            error_u = self.desire_u - u
            error_v = self.desire_v - v
            error_r = self.desire_r - r
            error_psi = ssa(self.desire_psi*np.pi/180.0 -psi)
            if abs(error_u) >0.1:
                error_u =error_u/abs(error_u) * 0.1
            if abs(error_r) >0.15:
                error_r =error_r/abs(error_r) * 0.1
            # PID controller
            tau_u = self.PID_u.output(error_u, None, self.dt)
            tau_v = self.PID_v.output(error_v, None, self.dt)
            tau_r = self.PID_r.output(error_r, None, self.dt)
            tau_psi = self.PID_psi.output(error_psi, None, self.dt)
            self.get_logger().info(f'tau_u Mode {tau_u}')
            pwm = saturation(self.pwm, self.pwm_limit[0], self.pwm_limit[1])

            # rpm_port    = saturation(self.rpm[0]+tau_r  , self.rpm_limit[0], self.rpm_limit[1])
            # rpm_stbd    = saturation(self.rpm[1]-tau_r+ tau_u, self.rpm_limit[0], self.rpm_limit[1])
            if pwm >1500:
                rpm_port    = saturation(self.rpm_port+tau_r  , self.rpm_limit[0], self.rpm_limit[1])
                rpm_stbd    = saturation(self.rpm_stbd-tau_r+tau_u, self.rpm_limit[0], self.rpm_limit[1])            
                rudder_port = saturation(0.0, self.rpm_limit[0], self.rpm_limit[1])
                rudder_stbd = saturation(0.0, self.rpm_limit[0], self.rpm_limit[1])
            else:
                rpm_port    = saturation(self.rpm_port+tau_r+tau_u  , self.rpm_limit[0], self.rpm_limit[1])
                rpm_stbd    = saturation(self.rpm_stbd-tau_r, self.rpm_limit[0], self.rpm_limit[1])            
                rudder_port = saturation(0.0, self.rpm_limit[0], self.rpm_limit[1])
                rudder_stbd = saturation(0.0, self.rpm_limit[0], self.rpm_limit[1])
        else:
            pwm = 1500.0
            rpm_port = 0.0
            rpm_stbd = 0.0
            rudder_port = 0.0
            rudder_stbd = 0.0



        # publish
        control_msg = Float64MultiArray()
        control_msg.data = [pwm, rpm_port, rpm_stbd, rudder_port, rudder_stbd]  
    
        self.control_publisher.publish(control_msg)

        # reset parameter
        self.navigation_msg = None

        self.rpm_port = rpm_port
        self.rpm_stbd = rpm_stbd
        self.prev_dt = system_time
        
def main(args=None):
    rclpy.init(args=args)
    try:
        control_node = Control()
        try:
            rclpy.spin(control_node)
        except KeyboardInterrupt:
            control_node.get_logger().info('Keyboard Interrupt (SIGINT)')
        finally:
            control_node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
