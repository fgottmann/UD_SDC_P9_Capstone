#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped,TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        
        # Publish waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.velocity = None
        self.stopline_wp_idx = -1
        self.base_waypoints = None
        self.waypoints_2d  = None
        self.waypoint_tree = None

        self.loop()
        
    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
        
            rate.sleep()
            
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        
        # Check if closest is ahead or behind
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]
        
        # Equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect - prev_vect,pos_vect-cl_vect)
        
        if val > 0:
            closest_idx = (closest_idx + 1)% len(self.waypoints_2d)
        return  closest_idx
    
    def decelerate_waypoints(self, waypoints,closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            stop_idx = min(len(waypoints)-1,max(self.stopline_wp_idx - closest_idx,0))
            dist = max(0.0,self.distance(waypoints,i,stop_idx) -2.5) # brake before the stop point to maintain asafety gap
            vel = math.sqrt(2.0 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            p.yaw = wp.yaw
            p.curvature = wp.curvature
            p.distance = wp.distance
            
            temp.append(p)
            
        return temp
    
    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        
        closest_idx = self.get_closest_waypoint_idx()
        points = self.base_waypoints.waypoints[closest_idx:(closest_idx+LOOKAHEAD_WPS)]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= closest_idx + LOOKAHEAD_WPS):
            lane.waypoints = points
        else:
            lane.waypoints = self.decelerate_waypoints(points, closest_idx)
            
         # calculate acceleration
        if len(lane.waypoints) > 1:
            for i,wp in enumerate(lane.waypoints):
                if i == 0:
                    d_v = (lane.waypoints[i+1].twist.twist.linear.x - lane.waypoints[i].twist.twist.linear.x)/np.maximum(0.001,lane.waypoints[i+1].distance - lane.waypoints[i].distance)
                    v_mean = 0.5*(lane.waypoints[i+1].twist.twist.linear.x + lane.waypoints[i].twist.twist.linear.x)
                    lane.waypoints[i].acceleration_x = d_v * np.maximum(0.2,v_mean)
                elif i == len(lane.waypoints) - 1:
                    d_v = (lane.waypoints[i].twist.twist.linear.x - lane.waypoints[i-1].twist.twist.linear.x)/np.maximum(0.001,lane.waypoints[i].distance - lane.waypoints[i-1].distance)
                    v_mean = 0.5*(lane.waypoints[i].twist.twist.linear.x + lane.waypoints[i-1].twist.twist.linear.x)
                    lane.waypoints[i].acceleration_x = d_v * np.maximum(0.2,v_mean)
                else:
                    d_v = (lane.waypoints[i+1].twist.twist.linear.x - lane.waypoints[i-1].twist.twist.linear.x)/np.maximum(0.001,lane.waypoints[i+1].distance - lane.waypoints[i-1].distance)
                    v_mean = 0.5*(lane.waypoints[i+1].twist.twist.linear.x + lane.waypoints[i-1].twist.twist.linear.x)
                    lane.waypoints[i].acceleration_x = d_v * np.maximum(0.2,v_mean)
        else:
            lane.waypoints[0].acceleration_x = 0
        
        return lane    
        
    def publish_waypoints(self,closest_idx):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        self.pose = msg
        
    def velocity_cb(self, msg):
        self.velocity = msg

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.base_waypoints = waypoints
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

