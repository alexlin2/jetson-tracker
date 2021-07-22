#!/usr/bin/env python3
from cone_tracker import TrackedTarget, WaypointGate
from cone_detection import ConeDetector
from yolov5 import YOLOv5

import rospy
import message_filters

from time import time
import numpy as np
import random 
import cv2

from sensor_msgs.msg import Image

model_path = "/home/alexlin/catkin_ws/src/jetson-tracker/src/best.pt"
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
        if len(self.tracked_targets) == 0:
            detected = self.detector.get_detection()
            coord_list = self.detector.get_cone_coordinates(detected)
            for bbox, point in zip(detected, coord_list):
                self.tracked_targets.append(TrackedTarget(random.randint(1, 101), bbox, point, self.detector.rgb_frame))

        for target in self.tracked_targets:
            if not target.update(self.detector.rgb_frame, self.detector.get_cone_coordinate):
                self.tracked_targets.remove(target)
                del(target)
                

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