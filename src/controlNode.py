#!/usr/bin/env python3
from coneTracker import TrackedTarget, WaypointGate
from coneDetection import ConeDetector
from visualization_msgs.msg import Marker, MarkerArray
from yolov5 import YOLOv5

import rospy
import message_filters

from time import time
import numpy as np
import random 
import cv2

from sensor_msgs.msg import Image

model_path = "/home/alex/catkin_ws/src/jetson-tracker/src/best.pt"
device = 'cuda'
net = YOLOv5(model_path, device)

class Controller:

    def __init__(self, detector) -> None:
        self.detector = detector
        self.tracked_targets = []

    def reset(self):
        pass
    
    def get_control(self):
        pass

    def update_tracking(self):
        self.detector.marker_array = MarkerArray()
        if len(self.tracked_targets) < 2:
            detected = self.detector.get_detection()
            coord_list = self.detector.get_cone_coordinates(detected)
            
            for bbox, point in zip(detected, coord_list):
                if self.check_target_validity(bbox):
                    self.tracked_targets.append(TrackedTarget(random.randint(1, 101), bbox, point, self.detector.rgb_frame))
            self.detector.marker_pub.publish(detector.marker_array)

        for target in self.tracked_targets:
            if not target.update(self.detector.rgb_frame, self.detector.get_cone_coordinate):
                self.tracked_targets.remove(target)
                del(target)
        
        self.detector.marker_pub.publish(self.detector.marker_array)

                
    def check_target_validity(self, bbox):
        tracked_arr = np.array([x.bbox[0] for x in self.tracked_targets])
        size_constrain = bbox[2]/bbox[3] < 1.0 and bbox[2]/bbox[3] > 0.6 and bbox[2]*bbox[3] < 20000
        if len(tracked_arr) > 0:
            if size_constrain and np.min(np.abs(tracked_arr - bbox[0])) > 30:
                return True
        else:
            return size_constrain

        return False

    def run(self):
        pass


def callback(rgb_frame, depth_frame, detector):
    detector.update(rgb_frame, depth_frame)
    

if __name__ == '__main__':

    rospy.init_node('controller', anonymous=True)

    detector = ConeDetector(net)
    controller = Controller(detector)
    detection_pub = rospy.Publisher("detected_image", Image, queue_size=1)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
    depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, detector)
    
    while not rospy.is_shutdown():
        if detector.start:
            controller.update_tracking()
            debug_frame = detector.bridge.cv2_to_imgmsg(detector.debug_frame, "rgb8")
            detection_pub.publish(debug_frame)