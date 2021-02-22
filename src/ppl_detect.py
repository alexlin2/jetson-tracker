#!/usr/bin/python3
"""
This file holds the PeopleDetection class that can be used to detect objects using a SSD model trained on COCO datasets.
Class has functions that can get 3d coordinates of detections and create Marker Spheres for visualizations
"""
from yolov5 import YOLOv5
from yolov5.utils.plots import color_list, plot_one_box
from yolov5.utils.general import xyxy2xywh
import random
import rospy
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

model_path = "/home/beast/yolov5/weights/yolov5s.pt"
device = "cuda"
colors = color_list()

class PeopleDetection:
    """ People Detection class with useful functions for getting coordinates of detections"""
    def __init__(self):
        self._net = YOLOv5(model_path, device) #jetson.inference.detectNet("ssd-mobilenet-v2")
        self.img = None
        self.width = None
        self.height = None
        self.need_cam_info = True
        self.camera_model = PinholeCameraModel()
        self.marker = Marker()
        self.marker_pub = rospy.Publisher("visualization_markers", Marker, queue_size=10)
        self.camera_info = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.info_callback)

    def get_detections(self, image):
        """
        Function that uses a SSD Mobilenet V2 model to run an inference on provided RGBA image at variable FPS
        :param image: RGBA image frame from realsense camera
        :return: List of detections found on the provided image and
        resulting image with bounding boxes, labels, and confidence %
        """
        self.img = image
        self.width = image.shape[1]
        self.height = image.shape[0]
        detections = self._net.predict(self.img)
        if detections.n > 0:
            for *xyxy, cond, cls in detections.pred[0]:
                if cls == 0:
                    label = f'{detections.names[int(cls)]} {cond:.2f}'
                    plot_one_box(xyxy, self.img, label=label, color=colors[int(cls)%10], line_thickness=3)
        return detections, self.img

    def get_person_coordinates(self, depth_image, detections):
        """
        Function that filters through detections and calculates the 3d coordinates of every person detected in list of
        detections
        :param depth_image: grayscale depth frame from realsense camera
        :param detections: list of detections found from rgb inference
        :return: list of coordinates of every person's coordinates
        """
        coord_list = []
        
        for *xywh, cond, cls in detections.xywh[0]:
            if cls == 0 and cond > 0.5:
                x,y,w,h = int(xywh[0]),int(xywh[1]),int(xywh[2]),int(xywh[3]),
                depth = depth_image[int(y), int(x)]
                depth_array = depth_image[int(y-h/5):int(y+h/10),int(x-w/4):int(x+w/4)].flatten()
                depth = np.median(depth_array[np.nonzero(depth_array)]) / 1000
                person_coord = self._get_coord(depth, x, y)
                #print(f'{x} {y} {depth:.2f}')
                self.marker = self.make_marker(person_coord)
                coord_list.append(person_coord)

        self.marker_pub.publish(self.marker)
        return coord_list

    def _get_coord(self, person_depth, x, y):
        """
        Helper function to calculate 3d coordinates using image_geometry package given pixel of person detected and
        respective depth value mapping
        :param person_depth: depth value at pixel representing center of person detected
        :param x: horizontal pixel value
        :param y: vertial pixel value
        :return: list of [x,y,z] of person relative to camera
        """
        unit_vector = self.camera_model.projectPixelTo3dRay((x, y))
        normalized_vector = [i / unit_vector[2] for i in unit_vector]
        point_3d = [j * person_depth for j in normalized_vector]
        return point_3d
    
    def make_marker(self, point_3d):
        """
        Function that creates Marker Spheres for people detected for visualization of people with respect to the camera
        and car (given camera is attached to car)
        Adds detections to a MarkerArray List
        :param point_3d: calcualted 3d point of person in image
        :param count: number of people detected
        """
        person_marker = Marker()
        person_marker.header.frame_id = "map"
        person_marker.ns = "person"
        person_marker.type = person_marker.SPHERE
        person_marker.action = person_marker.ADD
        person_marker.id = 0
        person_marker.pose.position.x = point_3d[0] 
        person_marker.pose.position.y = point_3d[2]
        person_marker.pose.position.z = point_3d[1]
        person_marker.pose.orientation.x = 0.0
        person_marker.pose.orientation.y = 0.0
        person_marker.pose.orientation.z = 0.0
        person_marker.pose.orientation.w = 1.0
        person_marker.scale.x = 1
        person_marker.scale.y = 1
        person_marker.scale.z = 1
        person_marker.color.a = 1.0
        person_marker.color.r = 0.0
        person_marker.color.g = 1.0
        person_marker.color.b = 0.0
        person_marker.lifetime = rospy.Duration(1)
        return person_marker

    def info_callback(self, info):
        """ Helper callback function for getting camera info for image_geometry package, only used one time"""
        if self.need_cam_info:
            print("got camera info")
            self.camera_model.fromCameraInfo(info)
            self.need_cam_info = False
