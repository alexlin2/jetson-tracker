#!/usr/bin/env python3
from coneTracker import get_cone_waypoints
from coneDetection import ConeDetector
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Twist, PointStamped, PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from yolov5 import YOLOv5
from scipy.spatial.transform import Rotation as R
from geographiclib.geodesic import Geodesic

import rospy
import message_filters

from time import time
import numpy as np
import random 
import math

from sensor_msgs.msg import Image

model_path = "/home/alexlin/catkin_ws/src/jetson-tracker/src/best.pt"
device = 'cuda'
net = YOLOv5(model_path, device)

def map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def get_distance_and_bearing(lat1, long1, lat2, long2):
    geodict = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)
    bearing = geodict['azi1'] 
    if bearing < 0: bearing += 360
    return geodict['s12'], bearing

def get_cone_coord(car_lat, car_long, camera_x, camera_y, heading):
    theta = np.deg2rad(heading)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    world_pos = np.matmul(np.linalg.inv(R), np.array([[camera_x, camera_y]]).T)
    dist = np.linalg.norm(world_pos,axis=0)
    angle = -np.rad2deg(np.arctan2(world_pos[1,0], world_pos[0,0]))+90
    geodict = Geodesic.WGS84.Direct(car_lat, car_long, angle, dist)
    
    return geodict['lat2'], geodict['lon2']

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

    def reset(self):
        pass
    
    def update_gps(self, gps_coord):
        self.gps_coord = gps_coord

    def update_heading(self, heading):
        self.heading = heading

    def get_control(self, waypoint):
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
                                                                            lat, long)
                self.reset_tracking = False
        else:
            if self.coneWaypoint.update(self.detector.rgb_frame, self.detector.get_cone_coordinate):
                lat, long = get_cone_coord(self.gps_coord.latitude, self.gps_coord.longitude, \
                                                        self.coneWaypoint.rel_point.point.x, self.coneWaypoint.rel_point.point.y, self.heading)
                dist_arr = np.zeros(len(self.cones))
                for idx, cone in enumerate(self.cones):
                    dist, _ = get_distance_and_bearing(lat, long, cone.gps_coord.latitude, cone.gps_coord.longitude)
                    dist_arr[idx] = dist
                print(dist_arr)
                if np.argmin(dist_arr) != self.cones.index(self.coneWaypoint):
                    self.reset_tracking = True
                waypoint_dist, waypoint_bearing = get_distance_and_bearing(self.gps_coord.latitude, self.gps_coord.longitude, \
                                                                        lat, long)
            else:
                self.reset_tracking = True
                self.coneWaypoint = None
                

        return self.coneWaypoint, waypoint_dist, waypoint_bearing

                
    def check_target_validity(self, bbox):
        return bbox[2]/bbox[3] < 1.0 and bbox[2]/bbox[3] > 0.6 and bbox[2]*bbox[3] < 30000

    def run(self):
        
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
    detector.update(rgb_frame, depth_frame)

def callback_gps(gps_coord, controller):
    controller.update_gps(gps_coord)
    
def callback_heading(heading, controller):
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
    compass_sub = rospy.Subscriber('mavros/global_position/compass_hdg', Float64, callback_heading, callback_args=controller, queue_size=1)
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