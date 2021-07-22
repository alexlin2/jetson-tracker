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


class TrackedTarget:

    def __init__(self, id, initBB, init_pos, frame):
        self.id = id
        self.bbox = initBB
        self.point = init_pos
        self.tracker = cv2.TrackerKCF_create()
        self.reset(frame, initBB)

    def __del__(self):
        print("person " + str(self.id) + " is out of frame")

    def reset(self, frame, initBB):
        x,y,w,h = initBB[0] - 0.5 * initBB[2], initBB[1] - 0.5 * initBB[3], initBB[2], initBB[3]
        self.tracker.init(frame, [int(x),int(y),int(w),int(h)])

    def update(self, frame, get_coord):
        success, box =self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.bbox = [x + 0.5 * w,y + 0.5 * h,w,h]
            update_coord = get_coord(self.bbox)
            print(update_coord)
            if cal_distance(update_coord, self.point) > 0.2:
                return False
            self.point = update_coord
        return success

class WaypointGate:

    def __init__(self) -> None:
        self.waypoint_pub = rospy.Publisher("/waypoint_next", PointStamped, queue_size=1)
        self.waypoint = PointStamped()

    def get_cone_coords(self, cones):
        waypoint = PointStamped()

        def midpoint(p1, p2):
            return (p1.point.x+p2.point.x)/2, (p1.point.y+p2.point.y)/2

        if len(cones) == 2:
            x, y = midpoint(cones[0], cones[1])
            self.waypoint.point.x = x
            self.waypoint.point.y = y