#!/usr/bin/env python3
from coneTracker import TrackedTarget, WaypointGate, add_pose_to_point, cal_distance
from coneDetection import ConeDetector
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import Twist, PointStamped, PoseStamped, PoseWithCovarianceStamped
from yolov5 import YOLOv5

import rospy
import message_filters

from time import time
import numpy as np
import random 

from sensor_msgs.msg import Image

model_path = "/home/alex/catkin_ws/src/jetson-tracker/src/best.pt"
device = 'cuda'
net = YOLOv5(model_path, device)

def map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def make_marker(self, point_3d):
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
    cone_marker.pose.position.y = point_3d[2]
    cone_marker.pose.position.z = point_3d[1]
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
        self.gps_pose = PoseStamped()
        self.reset_tracking = False

    def reset(self):
        pass
    
    def update_gps(self, gps_coord):
        self.gps_pose = gps_coord

    def get_control(self, waypoint):
        twist = Twist()
        x,y = waypoint.point.x, waypoint.point.y
        angular_z = np.arctan2(-x,y)
        linear_x = map(y, 1, 5, 0, 1)
        print(f"{angular_z:.2f} {linear_x:.2f}")
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        return twist

    def update_tracking(self):
        self.detector.marker_array = MarkerArray()
        if len(self.tracked_targets) < 2 or self.reset_tracking:
            detected = self.detector.get_detection()
            coord_list = self.detector.get_cone_coordinates(detected)
            
            for bbox, point in zip(detected, coord_list):
                if self.check_target_validity(bbox):
                    cone_point = add_pose_to_point(self.gps_pose, point)
                    for target in self.tracked_targets:
                        if cal_distance(cone_point, target.get_filtered_pose()) < 1.0:
                            target.reset(self.detector.rgb_frame, bbox)
                    else:
                        self.tracked_targets.append(TrackedTarget(random.randint(1, 101), bbox, point, self.gps_pose, self.detector.rgb_frame))
            self.reset_tracking = False       
            
        for target in self.tracked_targets:
            tracked, cone_coord = target.update(self.detector.rgb_frame, self.detector.get_cone_coordinate, self.gps_pose)
            self.reset_tracking = not tracked
            marker = make_marker(cone_coord)
            self.marker_array.markers.append(marker)
        

        self.detector.marker_pub.publish(self.detector.marker_array)

        return self.tracked_targets

                
    def check_target_validity(self, bbox):
        tracked_arr = np.array([x.bbox[0] for x in self.tracked_targets])
        size_constrain = bbox[2]/bbox[3] < 1.0 and bbox[2]/bbox[3] > 0.6 and bbox[2]*bbox[3] < 30000
        if len(tracked_arr) > 0:
            if size_constrain and np.min(np.abs(tracked_arr - bbox[0])) > 30:
                return True
        else:
            return size_constrain

        return False

    def run(self):
        
        cones = self.update_tracking()
        if len(cones) >= 2 and not self.waypoint_set:
            self.gate = WaypointGate(waypoint_pub)
            self.waypoint_set = True

        if len(cones) == 0:
            self.waypoint_set = False
            self.gate = None 

        if self.waypoint_set:
            waypoint = self.gate.get_waypoint_coords(cones)
            twist = self.get_control(waypoint)
            pub_command.publish(twist)
        


def callback(rgb_frame, depth_frame, gps_coord, detector, controller):
    controller.update_gps(gps_coord)
    detector.update(rgb_frame, depth_frame)
    

if __name__ == '__main__':

    rospy.init_node('controller', anonymous=True)

    detector = ConeDetector(net)
    controller = Controller(detector)
    detection_pub = rospy.Publisher("detected_image", Image, queue_size=1)
    pub_command = rospy.Publisher('/robot/navigation/input', Twist, queue_size=1)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    gps_sub = message_filters.Subscriber('/mavros/global_position/local', PoseWithCovarianceStamped, queue_size=1)
    waypoint_pub = rospy.Publisher("/waypoint_next", PointStamped, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, gps_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, detector, controller)
    
    while not rospy.is_shutdown():
        if detector.start:
            controller.run()
            debug_frame = detector.bridge.cv2_to_imgmsg(detector.debug_frame, "rgb8")
            detection_pub.publish(debug_frame)