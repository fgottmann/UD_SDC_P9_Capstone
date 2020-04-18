#!/usr/bin/env python

import os
import csv
import math

from geometry_msgs.msg import Quaternion

from styx_msgs.msg import Lane, Waypoint

import tf
import rospy
import numpy as np

CSV_HEADER = ['x', 'y', 'z', 'yaw']
MAX_ACCEL_X = 1.5
MAX_ACCEL_Y = 1.5
MAX_DECEL_END = 1.0


class WaypointLoader(object):

    def __init__(self):
        rospy.init_node('waypoint_loader', log_level=rospy.DEBUG)

        self.pub = rospy.Publisher('/base_waypoints', Lane, queue_size=1, latch=True)

        self.velocity = self.kmph2mps(rospy.get_param('~velocity'))
        self.new_waypoint_loader(rospy.get_param('~path'))
        rospy.spin()

    def new_waypoint_loader(self, path):
        if os.path.isfile(path):
            waypoints = self.load_waypoints(path)
            self.publish(waypoints)
            rospy.loginfo('Waypoint Loded')
        else:
            rospy.logerr('%s is not a file', path)

    def quaternion_from_yaw(self, yaw):
        return tf.transformations.quaternion_from_euler(0., 0., yaw)

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

    def load_waypoints(self, fname):
        waypoints = []
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            for i,wp in enumerate(reader):
                p = Waypoint()
                p.pose.pose.position.x = float(wp['x'])
                p.pose.pose.position.y = float(wp['y'])
                p.pose.pose.position.z = float(wp['z'])
                q = self.quaternion_from_yaw(float(wp['yaw']))
                p.pose.pose.orientation = Quaternion(*q)
                p.twist.twist.linear.x = float(self.velocity)
                # calculate distance vector
                if i == 0:
                    p.distance = 0
                else:
                    p.distance = waypoints[-1].distance + self.distance(p.pose.pose.position,waypoints[-1].pose.pose.position)
                    
                p.yaw = float(wp['yaw'])

                waypoints.append(p)
        
         # calculate curvature (requires at least 3 nodes)
        dd_x = 0;
        dd_y = 0;
        dd_x_prev = 0;
        dd_y_prev = 0;
        for i,wp in enumerate(waypoints):
            if i == 0:
                dd_x = np.cos(waypoints[i + 2].yaw) - np.cos(waypoints[i+1].yaw)/max(0.001,waypoints[i+2].distance - waypoints[i+1].distance)
                dd_y = np.sin(waypoints[i + 2].yaw) - np.sin(waypoints[i+1].yaw)/max(0.001,waypoints[i+2].distance - waypoints[i+1].distance)
                dd_x_prev = np.cos(waypoints[i + 1].yaw) - np.cos(waypoints[i].yaw)/max(0.001,waypoints[i+1].distance - waypoints[i].distance)
                dd_y_prev = np.sin(waypoints[i + 1].yaw) - np.sin(waypoints[i].yaw)/max(0.001,waypoints[i+1].distance - waypoints[i].distance)
                waypoints[i].curvature = np.cos(wp.yaw)*0.5*(dd_y + dd_y_prev) - np.sin(wp.yaw)*0.5*(dd_x + dd_x_prev)
            elif i == len(waypoints) - 1:
                waypoints[i].curvature = waypoints[i-1].curvature
            else:
                dd_x = np.cos(waypoints[i + 1].yaw) - np.cos(waypoints[i].yaw)/max(0.001,waypoints[i+1].distance - waypoints[i].distance)
                dd_y = np.sin(waypoints[i + 1].yaw) - np.sin(waypoints[i].yaw)/max(0.001,waypoints[i+1].distance - waypoints[i].distance)
                waypoints[i].curvature = np.cos(wp.yaw)*0.5*(dd_y + dd_y_prev) - np.sin(wp.yaw)*0.5*(dd_x + dd_x_prev)
                dd_x_prev = dd_x
                dd_y_prev = dd_y             
                
                
        waypoints = self.decelerateAtEnd(waypoints)
        waypoints = self.limit_velocity(waypoints) # limit velocity at end and during cornering
        
         # calculate acceleration
        for i,wp in enumerate(waypoints):
            if i == 0:
                d_v = (waypoints[i+1].twist.twist.linear.x - waypoints[i].twist.twist.linear.x)/max(0.001,waypoints[i+1].distance - waypoints[i].distance)
                v_mean = 0.5*(waypoints[i+1].twist.twist.linear.x - waypoints[i].twist.twist.linear.x)
                waypoints[i].acceleration_x = d_v / max(0.2,v_mean)
            elif i == len(waypoints) - 1:
                d_v = (waypoints[i].twist.twist.linear.x - waypoints[i-1].twist.twist.linear.x)/max(0.001,waypoints[i].distance - waypoints[i-1].distance)
                v_mean = 0.5*(waypoints[i].twist.twist.linear.x - waypoints[i-1].twist.twist.linear.x)
                waypoints[i].acceleration_x = d_v / max(0.2,v_mean)
            else:
                d_v = (waypoints[i+1].twist.twist.linear.x - waypoints[i-1].twist.twist.linear.x)/max(0.001,waypoints[i+1].distance - waypoints[i-1].distance)
                v_mean = 0.5*(waypoints[i+1].twist.twist.linear.x - waypoints[i-1].twist.twist.linear.x)
                waypoints[i].acceleration_x = d_v / max(0.2,v_mean)
                
        return waypoints

    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def decelerateAtEnd(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL_END * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints
    
    def limit_velocity(self,waypoints):
        for i,wp in enumerate(waypoints):
            waypoints[i].twist.twist.linear.x = np.minimum(waypoints[i].twist.twist.linear.x, np.sqrt(MAX_ACCEL_Y/max(1.0e-7,abs(waypoints[i].curvature))))
        
                
        return waypoints

    def publish(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointLoader()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint node.')
