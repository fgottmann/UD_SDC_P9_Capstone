#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from styx_msgs.msg import Lane, Waypoint
import math

from twist_controller import Controller

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `Controller` object
        # self.controller = Controller(<Arguments you wish to provide>)
        self.controller = Controller(vehicle_mass = vehicle_mass,
                                     fuel_capacity = fuel_capacity,
                                     brake_deadband = brake_deadband,
                                     decel_limit = decel_limit,
                                     accel_limit = accel_limit,
                                     wheel_radius = wheel_radius,
                                     wheel_base = wheel_base,
                                     steer_ratio = steer_ratio,
                                     max_lat_accel = max_lat_accel,
                                     max_steer_angle = max_steer_angle)        

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/vehicle/dbw_enabled',Bool,self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd',TwistStamped,self.req_cb)
        rospy.Subscriber('/current_velocity',TwistStamped,self.meas_cb)        
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)
        
        # local quantities
        self.current_vel = None
        self.current_angle_vel = None
        self.dbw_enabled = None
        self.target_vel = None
        self.target_angle_vel = None
        self.throttle = self.steering = self.brake = 0
        self.final_waypoints = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            if not None in (self.current_vel, self.current_angle_vel,
                             self.target_vel, self.target_angle_vel):
                self.throttle, self.brake,self.steering = self.controller.control(
                    self.current_vel,self.current_angle_vel,self.dbw_enabled,
                    self.target_vel, self.target_angle_vel)
                
                if self.dbw_enabled:
                    self.publish(self.throttle, self.brake, self.steering)
                
            rate.sleep()

    def dbw_enabled_cb(self,msg):
        self.dbw_enabled = msg
        
    def req_cb(self,msg):
        self.target_vel = msg.twist.linear.x
        self.target_angle_vel = msg.twist.angular.z
        
    def meas_cb(self,msg):
        self.current_vel = msg.twist.linear.x
        self.current_angle_vel = msg.twist.angular.z
    
    def final_waypoints_cb(self,msg):
        self.final_waypoints = msg
        
    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
