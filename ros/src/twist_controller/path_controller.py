from pid import PID
from lowpass import LowPassFilter
import rospy
import tf
import numpy as np
import scipy.linalg

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

USE_DERV_BASED_LQR = 1

class PathController(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
                
        # vel filter
        tau = 0.5
        ts = 0.02 # 50hz
        self.vel_lpf = LowPassFilter(tau,ts)
        
        
        # parameter
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        
        
        # controller parameter throttle
        kp = 0.3
        ki = 0.1
        kd = 0
        mn = 0
        mx = 0.25
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        # specify delays
        self.delay_lat = 0.15
        self.delay_lon = 0.2
           
        self.last_time = rospy.get_time()
        
        #specify control variables
        self.d = None
        self.s = None
        self.e_psi = None
        self.v = None
        self.ax = None
        self.ax_pred = None
        self.kappa = None
        self.kappa_pred = None
        self.valid = False
        self.last_steering = None
           
    def lqr_lat(self,v): 
        v = max(2.0,v)
        A = np.matrix([[0,v,0],
                       [0,0,v],
                       [0,0,0]])
        B = np.matrix([[0],
                       [0],
                       [1]])
        
        kappa_max = np.tan(self.max_steer_angle/self.steer_ratio)/self.wheel_base
        kappa_ref = 0.1*min(kappa_max,self.max_lat_accel/(max(3.0,v)**2))
        Q = np.matrix([[1.0/(1.0**2),0,0],
                       [0,1.0/((5.0*np.pi/180.0)**2),0],
                       [0,0,1.0/(kappa_ref**2)]])
        
        kappa_dot_ref = 0.5*self.max_lat_accel/(max(3.0,v)**2)
        R = np.matrix([[1.0/(kappa_dot_ref**2)]])
     
        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        K = -np.matrix(scipy.linalg.inv(R)*(B.T*X))
        return K
    
           
    def lqr_lat2(self,v): 
        v = max(2.0,v)
        A = np.matrix([[0,v],
                       [0,0]])
        B = np.matrix([[0],
                       [v]])
        
        Q = np.matrix([[1.0/(1.0**2),0],
                       [0,1.0/((5.0*np.pi/180.0)**2)]])
        
        kappa_max = np.tan(self.max_steer_angle/self.steer_ratio)/self.wheel_base
        kappa_ref = 0.1*min(kappa_max,self.max_lat_accel/(max(3.0,v)**2))
        R = np.matrix([[1.0/(kappa_ref**2)]])
     
        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
        K = -np.matrix(scipy.linalg.inv(R)*(B.T*X))
        return K
    
    def control(self, current_vel,current_pose, current_steering_angle, dbw_enabled,ref_path):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        current_time = rospy.get_time()
        if self.last_steering is None:
            self.last_steering = current_steering_angle
                
        if not dbw_enabled:
            self.last_time = current_time
            self.last_steering = current_steering_angle
            return 0.0, 0.0, 0.0
              
        self.localizeOnPath(current_pose,current_vel,ref_path)
        
        print("valid: {}, Nd: {}, s: {:f}, d: {:f}, e_psi: {:f}, curv: {:f}, curv_pre: {:f}, v: {}, ax: {:f}, ax_pre: {:f}".format(self.valid,len(ref_path.waypoints),self.s,self.d,self.e_psi,self.kappa,self.kappa_pred,self.v,self.ax,self.ax_pred))
  
        if self.valid == False:
            self.last_time = current_time
            self.last_steering = current_steering_angle
            return 0.0, 0.0, 0.0
        
        dt = min(0.03,current_time - self.last_time)
        
        v = 0.5*(self.v + np.abs(current_vel.twist.linear.x))
        K_lat = self.lqr_lat(v)
        K_lat = self.lqr_lat2(v)
        scaling = self.steer_ratio* (1.0 +(current_vel.twist.linear.x/30.0)**2) # + yaw gain
        last_curvature = np.tan(self.last_steering/scaling)/self.wheel_base
        if USE_DERV_BASED_LQR > 0: 
            K_lat = self.lqr_lat(v)
            x0 = np.matrix([[self.d],[self.e_psi],[last_curvature-self.kappa_pred]])
            new_curvature = dt*np.matmul(K_lat,x0) + last_curvature
        else:
            K_lat = self.lqr_lat2(v)
            x0 = np.matrix([[self.d],[self.e_psi]])
            new_curvature = np.matmul(K_lat,x0) + self.kappa_pred
        steering = max(-self.max_steer_angle,min(self.max_steer_angle, np.arctan(self.wheel_base * new_curvature) * scaling))
        self.last_steering = steering
       
        vel_error = self.v - v
        self.last_vel = v
        
        throttle = self.throttle_controller.step(vel_error, dt)
        brake = 0
        
        if self.v == 0 and v < 0.1:
            throttle = 0
            brake = 700
        
        elif throttle < 0.1 and (vel_error) < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = max(0,-decel)*self.vehicle_mass*self.wheel_radius
        
        return throttle, brake, steering
    
    def localizeOnPath(self,pose,velocity,ref_path):
        quaternion = (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w)
        euler_angles = tf.transformations.euler_from_quaternion(quaternion)
        if len(ref_path.waypoints) > 1:
            closest_point = None
            closest_dist  = None
            for i,wp in enumerate(ref_path.waypoints):
                dist = self.distance(pose.pose.position,wp.pose.pose.position)
                if None in (closest_point,closest_dist) or closest_dist > dist:
                    closest_dist = dist
                    closest_point = i
                    
            if closest_point is not None:
                if closest_point == len(ref_path.waypoints)-1:
                    closest_point = len(ref_path.waypoints)-2
                elif closest_point == 0:
                    closest_point = 0       
                else:
                    if self.distance(pose.pose.position,ref_path.waypoints[closest_point - 1].pose.pose.position) < \
                    self.distance(pose.pose.position,ref_path.waypoints[closest_point + 1].pose.pose.position):
                        closest_point -= 1
            else:
                self.valid = False
                return
            
            dx_path = ref_path.waypoints[closest_point+1].pose.pose.position.x - ref_path.waypoints[closest_point].pose.pose.position.x
            dy_path = ref_path.waypoints[closest_point+1].pose.pose.position.y - ref_path.waypoints[closest_point].pose.pose.position.y
            dist_total = max(1e-4,np.sqrt(dx_path*dx_path + dy_path*dy_path))
            dx_pose = pose.pose.position.x - ref_path.waypoints[closest_point].pose.pose.position.x
            dy_pose = pose.pose.position.y - ref_path.waypoints[closest_point].pose.pose.position.y
           
            proj_norm = (dx_pose*dx_path+dy_pose*dy_path)/(dist_total**2);
            proj_norm = max(0.0,min(1.0,proj_norm));
            proj_x = proj_norm*dx_path;
            proj_y = proj_norm*dy_path;
            
            self.s = ref_path.waypoints[closest_point].distance + proj_norm*(ref_path.waypoints[closest_point+1].distance -
                                                                              ref_path.waypoints[closest_point].distance)
            self.d = np.sign(dy_pose*dx_path- dx_pose*dy_path)*np.sqrt((proj_x-dx_pose)**2 + (proj_y-dy_pose)**2)
            yaw_path = np.arctan2(np.sin(ref_path.waypoints[closest_point].yaw) + proj_norm*(np.sin(ref_path.waypoints[closest_point + 1].yaw) -
                                                                                        np.sin(ref_path.waypoints[closest_point].yaw)),
                                np.cos(ref_path.waypoints[closest_point].yaw) + proj_norm*(np.cos(ref_path.waypoints[closest_point + 1].yaw) -
                                                                                        np.cos(ref_path.waypoints[closest_point].yaw)))

            self.e_psi = ((euler_angles[2] - yaw_path + np.pi) % (2.0*np.pi)) - np.pi # map to +/-pi
                        
            self.kappa = ref_path.waypoints[closest_point].curvature + proj_norm*(ref_path.waypoints[closest_point+1].curvature -
                                                                              ref_path.waypoints[closest_point].curvature)
            
            
            
            self.v = ref_path.waypoints[closest_point].twist.twist.linear.x + proj_norm*(ref_path.waypoints[closest_point+1].twist.twist.linear.x -
                                                                              ref_path.waypoints[closest_point].twist.twist.linear.x)  
            
            self.ax = ref_path.waypoints[closest_point].acceleration_x + proj_norm*(ref_path.waypoints[closest_point+1].acceleration_x -
                                                                              ref_path.waypoints[closest_point].acceleration_x)
            
            count = 1
            self.kappa_pred = self.kappa
            ind = closest_point + 1
            s_pred = self.s + max(0.0,self.delay_lat*velocity.twist.linear.x)*2.0
            while ind < len(ref_path.waypoints) and s_pred > ref_path.waypoints[ind].distance:
                self.kappa_pred += ref_path.waypoints[ind].curvature
                count += 1
                ind += 1
            self.kappa_pred /= count
            
            count = 1
            self.ax_pred = self.ax
            ind = closest_point + 1
            s_pred = self.s + max(0.0,self.delay_lon*velocity.twist.linear.x)*2
            while ind < len(ref_path.waypoints) and s_pred > ref_path.waypoints[ind].distance:
                self.ax_pred += ref_path.waypoints[ind].acceleration_x
                count += 1
                ind += 1
            self.ax_pred /= count
                            
        else:
            self.s = ref_path.waypoints[0].distance
            self.e_psi = ((euler_angles[2] - ref_path.waypoints[0].yaw + np.pi) % (2.0*np.pi)) - np.pi
            self.d = np.cos(ref_path.waypoints[0].yaw)*(pose.pose.position.y - ref_path.waypoints[0].pose.pose.position.y) - \
                     np.sin(ref_path.waypoints[0].yaw)*(pose.pose.position.x - ref_path.waypoints[0].pose.pose.position.x)
            self.kappa = ref_path.waypoints[0].curvature
            self.kappa_pred = ref_path.waypoints[0].curvature
            self.v = ref_path.waypoints[0].twist.twist.linear.x
            self.ax = ref_path.waypoints[0].acceleration_x
            self.ax_pred = ref_path.waypoints[0].acceleration_x
        
        # check path
        if not None in (self.e_psi,self.d):
            if np.abs(self.e_psi) < np.pi*0.5 and np.abs(self.d) < 5.0:
                self.valid = True
            else:
                self.valid = False
        else:
            self.valid = False
            
                
    
    
    
    def distance(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return np.sqrt(x*x + y*y + z*z)
    
    def distance_acc(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += distance(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
