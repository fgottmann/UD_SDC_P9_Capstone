from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class TwistController(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        # feedforward
        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1,max_lat_accel,max_steer_angle)
        
        # controller parameter throttle
        kp = 0.3
        ki = 0.1
        kd = 0
        mn = 0
        mx = 0.25
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
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
           
        self.last_time = rospy.get_time()

    def control(self, current_vel,current_angle_vel, dbw_enabled, target_vel, target_angle_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        current_time = rospy.get_time()
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.last_time = current_time
            return 0.0, 0.0, 0.0
        
        steering = self.yaw_controller.get_steering(target_vel,target_angle_vel,current_vel)
        
        vel_error = target_vel - current_vel
        self.last_vel = current_vel
        
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        if target_vel == 0 and current_vel < 0.1:
            throttle = 0
            brake = 700
        
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = max(0,-decel)*self.vehicle_mass*self.wheel_radius
        
        return throttle, brake, steering
