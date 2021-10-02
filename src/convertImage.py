#!/usr/bin/env python3

from cv_bridge import CvBridge

bridge = CvBridge()

def convert_rosimg_to_cv2(frame, conversion):
    return bridge.imgmsg_to_cv2(frame, conversion)

def convert_cv2_to_rosimg(frame, conversion):
    return bridge.cv2_to_imgmsg(frame, conversion)