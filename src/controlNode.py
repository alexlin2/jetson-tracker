#!/usr/bin/env python3
from coneTracker import TrackedTarget, add_pose_to_point, cal_distance
from coneDetection import ConeDetector
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Twist, PointStamped, PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import Odometry
from mavros_msgs.srv import CommandHome
from std_msgs.msg import Float64
from yolov5 import YOLOv5
from scipy.spatial.transform import Rotation as R


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

def make_marker(point_3d):
    """
    Function that creates Marker Spheres for people detected for visualization of people with respect to the camera
    and car (given camera is attached to car)
    Adds detections to a MarkerArray List
    :param point_3d: calcualted 3d point of cone in image
    :param count: number of people detected
    """
    cone_marker = Marker()
    cone_marker.header.frame_id = "map"
    cone_marker.ns = "cone" + str(np.random.randint(100))
    cone_marker.type = cone_marker.CYLINDER
    cone_marker.action = cone_marker.ADD
    cone_marker.id = 0
    cone_marker.pose.position.x = point_3d[0] 
    cone_marker.pose.position.y = point_3d[1]
    cone_marker.pose.position.z = point_3d[2]
    cone_marker.pose.orientation.x = 0.0
    cone_marker.pose.orientation.y = 0.0
    cone_marker.pose.orientation.z = 0.0
    cone_marker.pose.orientation.w = 1.0
    cone_marker.scale.x = 0.2
    cone_marker.scale.y = 0.2
    cone_marker.scale.z = 0.2
    cone_marker.color.a = 1.0
    cone_marker.color.r = 1.0
    cone_marker.color.g = 0.0
    cone_marker.color.b = 0.0
    cone_marker.lifetime = rospy.Duration(0.1)
    return cone_marker

class Controller:

    def __init__(self, detector) -> None:
        self.detector = detector
        self.waypoint_set = False
        self.tracked_targets = []
        self.marker_array = MarkerArray()
        self.marker_pub = rospy.Publisher("visualization_markers", MarkerArray, queue_size=1)
        self.gate = None
        self.gps_pose = PoseWithCovarianceStamped()
        self.heading = 0.0
        self.reset_tracking = False
        self.idx = 0

    def reset(self):
        pass
    
    def update_gps(self, gps_coord, heading):
        self.gps_pose = gps_coord
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
        self.marker_array = MarkerArray()
        if len(self.tracked_targets) < 1 or self.reset_tracking:
            detected = self.detector.get_detection()
            coord_list = self.detector.get_cone_coordinates(detected)
            
            for bbox, point in zip(detected, coord_list):
                tracked_points = [cal_distance(point, x.rel_point) for x in self.tracked_targets]
                if self.check_target_validity(bbox, tracked_points) or len(self.tracked_targets) == 0:
                    self.tracked_targets.append(TrackedTarget(self.idx, bbox, point, self.gps_pose, self.detector.rgb_frame))
                    self.idx += 1
                
            self.reset_tracking = False       
            
        for target in self.tracked_targets:
            tracked, rel_coord = target.update(self.detector.rgb_frame, self.detector.get_cone_coordinate, self.gps_pose)
            self.reset_tracking = not tracked
            print(target.id)
            if not tracked:
                self.reset_tracking = True
                target.frames_lost += 1
                if target.frames_lost > 15:
                    self.tracked_targets.remove(target)
                    del(target)

            marker = make_marker([rel_coord.point.x, rel_coord.point.y, rel_coord.point.z])
            self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)

        return self.tracked_targets
                
    def check_target_validity(self, bbox, tracked_points):
        size_constrain = bbox[2]/bbox[3] < 1.1 and bbox[2]/bbox[3] > 0.5 and bbox[2]*bbox[3] < 30000
        if len(tracked_points) > 0:
            if size_constrain and np.min(tracked_points) > 0.5:
                return True
        else:
            return size_constrain

        return False

    def run(self):
        
        cones = self.update_tracking()

        if len(cones) > 0:
            waypoint = cones[0].rel_point
            #print(waypoint)
            twist = self.get_control(waypoint)
            pub_command.publish(twist)
        
        else:
            t = Twist
            pub_command.publish(t)


def callback(rgb_frame, depth_frame, detector):
    detector.update(rgb_frame, depth_frame)

def callback_2(gps_coord, heading, controller):
    controller.update_gps(gps_coord.pose, heading)
    pub_pose.publish(gps_coord.pose)
    

if __name__ == '__main__':

    rospy.init_node('controller', anonymous=True)
    print("node initialized")

    detector = ConeDetector(net)
    controller = Controller(detector)
    detection_pub = rospy.Publisher("detected_image", Image, queue_size=1)
    pub_command = rospy.Publisher('/robot/navigation/input', Twist, queue_size=1)
    pub_pose = rospy.Publisher('gps/pose', PoseWithCovarianceStamped, queue_size=1)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    gps_sub = message_filters.Subscriber('/mavros/global_position/local', Odometry, queue_size=1)
    compass_sub = message_filters.Subscriber('mavros/global_position/compass_hdg', Float64)
    # set_home = rospy.ServiceProxy("/mavros/cmd/set_home", CommandHome)
    # if set_home(yaw=0.0, latitude= 0.0,longitude= 0.0, altitude= 0.0):
    #     print("Correct home point set!")
    # else: 
    #     print("Home point set failed!")
    waypoint_pub = rospy.Publisher("/waypoint_next", PointStamped, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, detector)
    ts_2 = message_filters.ApproximateTimeSynchronizer([gps_sub, compass_sub], 10, 5, allow_headerless=True)
    ts_2.registerCallback(callback_2, controller)
    print("Setup complete!")
    
    while not rospy.is_shutdown():
        if detector.start:
            controller.run()
            debug_frame = detector.bridge.cv2_to_imgmsg(detector.debug_frame, "rgb8")
            detection_pub.publish(debug_frame)