########################################
#	Vehicle localization from landmark recognition
#
# 	Matthew (MJ) Moscola & Zhengzhi (Alex) Lin
#	September, 2021
#   
# 	The Vehicle Coordinate system has its origin at the vehicle centroid. The X-axis points forward, Y 
#	to the left, and Z up.
#	The origin of the World Coordinate system is set arbitrarily. The X-axis points east, Y north, and Z 
#	up.
#	The Camera Coordinate (CC) system has its origin at the focal point of the camera. The Z-axis is
#	depth. In the image, the X-axis points right and the y-axis down.
#	Distances in all coordinate systems are in meters.
#
#	A vehicle finds the distance and bearing to a landmark, whose location is known.
#	Based on the position of the landmark, the vehicle then updates its own estimated position.
#	We use large cones at known positions as a stand-in for landmarks. There are assumed to be no 
#	stray cones on the course.
########################################

#!/usr/bin/env python3

# import functions from coneTracker, coneDetection, and GPSfindCone
from coneTracker import get_cone_waypoints
from coneDetection import ConeDetector
from GPSfindCone import get_cone_coord, get_distance_and_bearing

# import ROS messages to communicate with the pixhawk
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist, PointStamped, PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
import rospy
import message_filters

# import YOLO to detect cones
from yolov5 import YOLOv5

# import linear algebra and GPS libraries
from scipy.spatial.transform import Rotation as R
from geographiclib.geodesic import Geodesic

# import system libraries
from time import time
import numpy as np
import random 
import math


# specify where the cone detection model is
model_path = "/home/alexlin/catkin_ws/src/jetson-tracker/src/best.pt"
device = 'cuda'
net = YOLOv5(model_path, device)


def map(x, in_min, in_max, out_min, out_max):
    ######################################
    '''
    linearly scales and offsets an input value

    Input:    x: data, 
              in_min: minimum of input value,
              in_max: maximum of input value,
              out_min: minimum of output value,
              out_max: minimum of output value,
          
    Output:   transformed value of input value

    '''
    ######################################
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# Main Controller that integrates cone detection, cone tracking and vehicle controller
class Controller:

    def __init__(self, detector, cones, longlat_arr) -> None:
        self.detector = detector
        self.waypoint_set = False
        self.coneWaypoint = None
        self.marker_array = MarkerArray()
        self.gps_coord = NavSatFix()
        self.heading = 0.0
        self.reset_tracking = False
        self.cones = cones
        self.longlat_arr = longlat_arr
        self.heading_hist = deque(maxlen=10)
    
    def update_gps(self, gps_coord):
        ######################################
        '''
        updates the gps coordinate of the car

        Input: gps_coord: current gps coordinate of car

        '''
        ######################################
        _, bearing = get_distance_and_bearing(self.gps_coord.latitude, self.gps_coord.longitude, gps_coord.latitude, gps_coord.longitude)
        self.heading_hist.appendleft(bearing)
        self.heading = np.mean(self.heading_hist)
        self.gps_coord = gps_coord

    def update_heading(self, heading):
        ######################################
        '''
        updates the compass heading of the car

        Input: heading: current heading of car

        '''
        ######################################
        self.heading = heading

    def get_control(self, waypoint):
        ######################################
        '''
        Returns the control messages that gets sent to the car

        Input: gps_coord: waypoint coordinate of the next cone

        Output: ROS twist message that contains the steering and throttle value to control the car
                
        '''
        ######################################
        twist = Twist()
        x,y = waypoint.point.x, waypoint.point.y
        if not math.isnan(x) and not math.isnan(y):
            angular_z = np.arctan2(-x,y) 
            linear_x = map(y, 0, 5, 0, 1)
        else:
            angular_z = 0.0
            linear_x = 0.0
        #print(f"{angular_z:.2f} {linear_x:.2f}")
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        return twist

    def update_tracking(self):
        ######################################
        '''
        Returns the cone that the car is currently tracking, and the distance and bearing to the cone

        Output: current cone object, 
                distance to the current tracked cone,
                bearing to the current tracked cone

        '''
        ######################################
        waypoint_dist = 0
        waypoint_bearing = 0
        if self.coneWaypoint is None or self.reset_tracking:
            detected = self.detector.get_detection()
            coord_list = self.detector.get_cone_coordinates(detected)
            if len(detected) > 0:
                point_arr = np.array([np.array([cone.point.x, cone.point.y]) for cone in coord_list])
                closest_cone_idx = np.argmin(np.linalg.norm(point_arr))    
                closest_cone = coord_list[closest_cone_idx]
                dist_arr = np.zeros(len(self.cones))
                lat, long = get_cone_coord(self.gps_coord.latitude, self.gps_coord.longitude, closest_cone.point.x, closest_cone.point.y, self.heading)
                for idx, cone in enumerate(self.cones):
                    dist, _ = get_distance_and_bearing(lat, long, cone.gps_coord.latitude, cone.gps_coord.longitude)
                    dist_arr[idx] = dist
                self.coneWaypoint = self.cones[np.argmin(dist_arr)]
                self.coneWaypoint.reset(self.detector.rgb_frame, detected[closest_cone_idx], self.detector.get_cone_coordinate)
                waypoint_dist, waypoint_bearing = get_distance_and_bearing(self.gps_coord.latitude, self.gps_coord.longitude, \
                                                                            lat,  long)
                self.reset_tracking = False
        else:
            if self.coneWaypoint.update(self.detector.rgb_frame, self.detector.get_cone_coordinate):
                lat, long = get_cone_coord(self.gps_coord.latitude, self.gps_coord.longitude, \
                                                        self.coneWaypoint.rel_point.point.x, self.coneWaypoint.rel_point.point.y, self.heading)
                dist_arr = np.zeros(len(self.cones))
                #print(self.heading)
                for idx, cone in enumerate(self.cones):
                    dist, _ = get_distance_and_bearing(lat, long, cone.gps_coord.latitude, cone.gps_coord.longitude)
                    dist_arr[idx] = dist
                #print(dist_arr)
                if np.argmin(dist_arr) != self.cones.index(self.coneWaypoint) or np.argmin(dist_arr) > 10.0:
                    self.reset_tracking = True
                waypoint_dist, waypoint_bearing = get_distance_and_bearing(self.gps_coord.latitude, self.gps_coord.longitude, \
                                                                        lat,  long)
            else:
                self.reset_tracking = True
                self.coneWaypoint = None
                

        return self.coneWaypoint, waypoint_dist, waypoint_bearing

                
    def check_target_validity(self, bbox):
        ######################################
        '''
        determines the validity of the tracked bounding box by calculating the size and ratio of height / width

        @params bbox: bounding box of cone seen in the RGB frame

        @return True | False 
       
        '''
        ######################################
        return bbox[2]/bbox[3] < 1.0 and bbox[2]/bbox[3] > 0.6 and bbox[2]*bbox[3] < 30000

    def run(self):
        ######################################
        '''
        main control loop 

        Still work in progress
                
        '''
        ######################################
        cone, dist, bearing = self.update_tracking()
        
        if cone is not None:
            print("distance: " + str(dist) + " bearing: " + str(bearing) + " id: " + cone.id)
        #     waypoint = cone.rel_point
        #     #print(waypoint)
        #     twist = self.get_control(waypoint)
        #     pub_command.publish(twist)
        
        # else:
        #     t = Twist
        #     pub_command.publish(t)


def callback(rgb_frame, depth_frame, detector):
    '''
    callback function that updates the rgb and depth frame  

    '''
    detector.update(rgb_frame, depth_frame)

def callback_gps(gps_coord, controller):
    '''
    callback function that updates the gps coordinate
       
    '''
    controller.update_gps(gps_coord)
    
def callback_heading(heading, controller):
    '''
    callback function that updates the heading
       
    '''
    controller.update_heading(heading)
        

if __name__ == '__main__':

    rospy.init_node('controller', anonymous=True)
    print("node initialized")

    cones, longlat_arr = get_cone_waypoints("/home/alexlin/catkin_ws/src/jetson-tracker/data.csv")

    detector = ConeDetector(net)
    controller = Controller(detector, cones, longlat_arr)
    detection_pub = rospy.Publisher("detected_image", Image, queue_size=1)
    pub_command = rospy.Publisher('/robot/navigation/input', Twist, queue_size=1)
    #pub_pose = rospy.Publisher('gps/pose', PoseWithCovarianceStamped, queue_size=1)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    gps_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, callback_gps, callback_args=controller, queue_size=1)
    #compass_sub = rospy.Subscriber('mavros/global_position/compass_hdg', Float64, callback_heading, callback_args=controller, queue_size=1)
    # set_home = rospy.ServiceProxy("/mavros/cmd/set_home", CommandHome)
    # if set_home(yaw=0.0, latitude= 0.0,longitude= 0.0, altitude= 0.0):
    #     print("Correct home point set!")
    # else: 
    #     print("Home point set failed!")
    waypoint_pub = rospy.Publisher("/waypoint_next", PointStamped, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, detector)

    print("Setup complete!")
    
    while not rospy.is_shutdown():
        if detector.start:
            #print("detector started!")
            controller.run()
            debug_frame = detector.bridge.cv2_to_imgmsg(detector.debug_frame, "rgb8")
            detection_pub.publish(debug_frame)