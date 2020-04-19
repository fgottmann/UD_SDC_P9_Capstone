#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped,TwistStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import numpy as np
import tf
import cv2
import yaml
import math
import threading
import time

STATE_COUNT_THRESHOLD = 1

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.dist_light = 0
        self.output_count = 0
        self.save_count = 0
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_info = None
        self.velocity = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub4 = rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        sub7 = rospy.Subscriber('/camera_info', CameraInfo, self.camera_info_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.image_lock = threading.RLock()

        
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        
        rospy.spin()

    def camera_info_cb(self,msg):
        self.camera_info = msg
        
    def pose_cb(self, msg):
        self.pose = msg
        
    def velocity_cb(self, msg):
        self.velocity = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        if not self.waypoints_2d:
            self.base_waypoints = waypoints
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        if self.image_lock.acquire(blocking = False):
            self.has_image = True
            self.camera_image = msg
            if self.pose and self.velocity and self.waypoints_2d and self.waypoint_tree and self.output_count >= 0:
                light_wp, state = self.process_traffic_lights()
                self.output_count = 0
            else:
                self.output_count += 1
                self.image_lock.release()
                return
            
    
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                
            self.state_count += 1
            
            self.image_lock.release()
            

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state
        if not self.has_image:
            self.prev_light_loc = None
            return 0
        if not hasattr(self,'light_classifier'):
            return 0
 
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        
        if self.camera_info:
            cv_image_undst = cv2.undistort(cv_image, self.camera_info.K, self.camera_info.D, None, self.camera_info.K)
        else:
            cv_image_undst = np.copy(cv_image)
 
#          
#             name = "TL_{:06d}_{:04d}_{}.jpg".format(int(self.save_count),int(self.dist_light),state)
#             self.save_count += 1
#             cv2.imwrite("images/" + name, cv_image_undst)

        #Get classification
        return self.light_classifier.get_classification(cv_image_undst)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints_2d)
            for i, light in enumerate(self.lights):
                #get stop line index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                # find closest stop line index
                d = temp_wp_idx - car_wp_idx
                if d >= -1 and d < diff:
                    diff = d 
                    closest_light = light
                    line_wp_idx = temp_wp_idx
            
            if closest_light:
                self.dist_light = self.distance(self.base_waypoints.waypoints,car_wp_idx,line_wp_idx)
        
        a_stop = 1.0
        stop_t = np.abs(self.velocity.twist.linear.x)/a_stop
        stop_distance = np.abs(self.velocity.twist.linear.x) - 0.5*a_stop*(stop_t**2)
        if closest_light and self.dist_light <= max(20.0,max(4.0*np.abs(self.velocity.twist.linear.x),1.5*stop_distance + 1.0*np.abs(self.velocity.twist.linear.x))): # only return waypoint if its close
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
        return -1, TrafficLight.UNKNOWN
    
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
