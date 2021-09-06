import cv2
import numpy as np
from geometry_msgs.msg import PointStamped, Point
import rospy
from random import random
import math
from collections import deque

def add_pose_to_point(pose, point):
    abs_x = pose.pose.pose.position.x
    abs_y = pose.pose.pose.position.y
    rel_x = point.point.x
    rel_y = point.point.y
    d = np.linalg.norm(np.array([rel_x, rel_y]))
    theta = np.arctan2(-rel_x, rel_y)
    return PointStamped(point=Point(pose.pose.pose.position.x + point.point.x, pose.pose.pose.position.y + point.point.y, 0))

def cal_distance(point_x, point_y):
    dx = abs(point_x.point.x - point_y.point.x)
    dy = abs(point_x.point.y - point_y.point.y)
    return np.hypot(dx, dy)


class TrackedTarget:

    def __init__(self, id, initBB, init_pos, init_gps_pose, frame):
        self.id = id
        self.bbox = initBB
        self.rel_point = init_pos
        self.abs_point = add_pose_to_point(init_gps_pose, init_pos)
        self.point_hist = deque(maxlen=50)
        self.tracker = cv2.TrackerKCF_create()
        self.reset(frame, initBB)
        self.frames_lost = 0

    def __del__(self):
        print("Cone " + str(self.id) + " is out of frame")

    def reset(self, frame, initBB):
        x,y,w,h = initBB[0] - 0.5 * initBB[2], initBB[1] - 0.5 * initBB[3], initBB[2], initBB[3]
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, [int(x),int(y),int(w),int(h)])

    def update(self, frame, get_rel_coord, gps_pose):
        success, box =self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.bbox = [x + 0.5 * w,y + 0.5 * h,w,h]
            updated_rel_point = get_rel_coord(self.bbox)
            if cal_distance(updated_rel_point, self.rel_point) > 0.5:
                return False, self.abs_point
            self.rel_point = updated_rel_point
            self.abs_point = add_pose_to_point(gps_pose, updated_rel_point)
            self.point_hist.appendleft(self.abs_point)
        return success, self.rel_point


