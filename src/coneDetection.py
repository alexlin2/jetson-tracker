#!/usr/bin/env python3
from yolov5.utils.plots import colors, plot_one_box
import random
import rospy
import message_filters
import cv2

import numpy as np
from time import time
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ConeDetector:

    def __init__(self, net, debug = True):
        self.bridge = CvBridge()
        self.debug = debug
        self.net = net
        self.start= False
        self.rgb_frame = None
        self.depth_frame = None
        self.debug_frame = None
        self.need_cam_info = True
        self.camera_model = PinholeCameraModel()
        self.marker_pub = rospy.Publisher("visualization_markers", MarkerArray, queue_size=1)
        self.camera_info = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.info_callback)
        self.detected_targets = []
        self.marker_array = MarkerArray()

    def get_detection(self):
        detections = self.net.predict(self.rgb_frame)
        detected = []
        for bbox in detections.pred:
            for *xyxy, cond, cls in bbox:
                if cls == 0 and cond > 0.5:
                    x,y,w,h = int(xyxy[0]),int(xyxy[1]),int(xyxy[2] - xyxy[0]),int(xyxy[3]-xyxy[1])
                    if self.debug:
                        label = f'{detections.names[int(cls)]} {cond:.2f}'
                        plot_one_box(xyxy, self.debug_frame, label=label, color=colors(int(cls),True), line_thickness=3)
                    detected.append([x + 0.5 * w,y + 0.5 * h,w,h])
        self.detected_targets = detected
        return detected

    def get_cone_coordinate(self, xywh):
        x,y,w,h = xywh[0], xywh[1], xywh[2], xywh[3]
        depth = self.depth_frame[int(y), int(x)]
        depth_array = self.depth_frame[int(y-h/10):int(y+h/5),int(x-w/4):int(x+w/4)].flatten()
        depth = np.median(depth_array[np.nonzero(depth_array)]) / 1000
        cone_coord = self._get_coord(depth, x, y)
        marker = self.make_marker(cone_coord)
        self.marker_array.markers.append(marker)
        cone = PointStamped()
        cone.point.x, cone.point.y, cone.point.z = cone_coord[0], cone_coord[2], cone_coord[1]

        cv2.rectangle(self.debug_frame, (int(x-0.5*w), int(y-0.5*h)), (int(x + 0.5*w), int(y + 0.5*h)),
				(0, 255, 255), 2)

        return cone

    def get_cone_coordinates(self, targets):
        """
        Function that filters through detections and calculates the 3d coordinates of every cone detected in list of
        detections
        :param depth_image: grayscale depth frame from realsense camera
        :param detections: list of detections found from rgb inference
        :return: list of coordinates of every cone's coordinates
        """
        cone_list = []
        
        for xywh in targets:
            x,y,w,h = xywh[0], xywh[1], xywh[2], xywh[3]
            depth = self.depth_frame[int(y), int(x)]
            depth_array = self.depth_frame[int(y-h/10):int(y+h/5),int(x-w/4):int(x+w/4)].flatten()
            cv2.rectangle(self.debug_frame, (int(x-w/4), int(y-h/10)), (int(x+w/4), int(y+h/5)),
				(0, 255, 255), 2)
            depth = np.median(depth_array[np.nonzero(depth_array)]) / 1000
            cone_coord = self._get_coord(depth, x, y)
            #print(f'{x} {y} {depth:.2f}')
            marker = self.make_marker(cone_coord)
            self.marker_array.markers.append(marker)
            cone = PointStamped()
            cone.point.x, cone.point.y, cone.point.z = cone_coord[0], cone_coord[2], cone_coord[1]
            cone_list.append(cone)

        return cone_list

    def _get_coord(self, cone_depth, x, y):
        """
        Helper function to calculate 3d coordinates using image_geometry package given pixel of cone detected and
        respective depth value mapping
        :param cone_depth: depth value at pixel representing center of cone detected
        :param x: horizontal pixel value
        :param y: vertial pixel value
        :return: list of [x,y,z] of cone relative to camera
        """
        unit_vector = self.camera_model.projectPixelTo3dRay((x, y))
        normalized_vector = [i / unit_vector[2] for i in unit_vector]
        point_3d = [j * cone_depth for j in normalized_vector]
        return point_3d

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

    def update(self, rgb_frame, depth_frame):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(rgb_frame, "rgb8")
        # depth_frame_16 = self.bridge.imgmsg_to_cv2(depth_frame, "passthrough")
        # df_dp = np.expand_dims(depth_frame_16, axis=-1).astype(np.uint8)
        # df_dp = np.tile(df_dp, (1, 1, 3))
        # self.debug_frame = cv2.cvtColor(df_dp, cv2.COLOR_RGB2RGBA)
        self.debug_frame = self.bridge.imgmsg_to_cv2(rgb_frame, "rgb8")
        self.depth_frame = self.bridge.imgmsg_to_cv2(depth_frame, "passthrough")
        self.start = True

    def info_callback(self, info):
        """ Helper callback function for getting camera info for image_geometry package, only used one time"""
        if self.need_cam_info:
            print("got camera info")
            self.camera_model.fromCameraInfo(info)
            self.need_cam_info = False
