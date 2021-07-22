import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
import rospy
from random import random
import math

def cal_distance(point_x, point_y):
    dx = abs(point_x.point.x - point_y.point.x)
    dy = abs(point_x.point.y - point_y.point.y)
    return np.hypot(dx, dy)

def midpoint(p1, p2):
    return (p1.point.x+p2.point.x)/2, (p1.point.y+p2.point.y)/2

class TrackedTarget:

    def __init__(self, id, initBB, init_pos, frame):
        self.id = id
        self.bbox = initBB
        self.point = init_pos
        self.tracker = cv2.TrackerKCF_create()
        self.reset(frame, initBB)

    def __del__(self):
        print("Cone " + str(self.id) + " is out of frame")

    def reset(self, frame, initBB):
        x,y,w,h = initBB[0] - 0.5 * initBB[2], initBB[1] - 0.5 * initBB[3], initBB[2], initBB[3]
        self.tracker.init(frame, [int(x),int(y),int(w),int(h)])

    def update(self, frame, get_coord):
        success, box =self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.bbox = [x + 0.5 * w,y + 0.5 * h,w,h]
            update_coord = get_coord(self.bbox)
            if cal_distance(update_coord, self.point) > 0.2:
                return False
            self.point = update_coord
        return success

class WaypointGate:

    def __init__(self, waypoint_pub) -> None:
        self.waypoint_pub = waypoint_pub
        self.waypoint = PointStamped()
        self.rel_pos = {}

    def get_waypoint_coords(self, cones):

        if len(cones) >= 2:
            cone_indice = np.argsort(np.array([cone.point.point.y for cone in cones]))[:2]
            x, y = midpoint(cones[cone_indice[0]].point, cones[cone_indice[1]].point)
            self.rel_pos[cones[cone_indice[0]].id] = np.array([cones[cone_indice[0]].point.point.x - x, cones[cone_indice[0]].point.point.y - y])
            self.rel_pos[cones[cone_indice[1]].id] = np.array([cones[cone_indice[1]].point.point.x - x, cones[cone_indice[1]].point.point.y - y])
            self.waypoint.point.x = x
            self.waypoint.point.y = y
            self.waypoint.header.frame_id = "map"
            self.waypoint_pub.publish(self.waypoint)

        if len(cones) == 1:
            cone = cones[0]
            if self.rel_pos.get(cone.id) is not None:
                rel_pos = self.rel_pos.get(cone.id)
                self.waypoint.point.x = cone.point.point.x - rel_pos[0]
                self.waypoint.point.y = cone.point.point.y - rel_pos[1]
                self.waypoint.header.frame_id = "map"
                self.waypoint_pub.publish(self.waypoint)
        
        return self.waypoint