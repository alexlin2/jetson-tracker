########################################
#	Vehicle localization from landmark recognition
#
# 	Matthew (MJ) Moscola & Zhengzhi (Alex) Lin
#	September, 2021
#   
# 	The Vehicle Coordinate system has its origin at the vehicle centroid. The X-axis points forward, Y 
#	to the left, and Z up.
#	The origin of the World Coordinate system is set arbitrarily. The X-axis points east, Y north, and Z 
#	up.
#	The Camera Coordinate (CC) system has its origin at the focal point of the camera. The Z-axis is
#	depth. In the image, the X-axis points right and the y-axis down.
#	Distances in all coordinate systems are in meters.
#
#	Contains the definition of the cone object that holds information about the location of the cone

########################################

#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import NavSatFix
from GPSfindCone import get_cone_coord, get_distance_and_bearing


def add_pose_to_point(pose, point):
    abs_x = pose.pose.position.x
    abs_y = pose.pose.position.y
    rel_x = point.point.x
    rel_y = point.point.y
    d = np.linalg.norm(np.array([rel_x, rel_y]))
    theta = np.arctan2(-rel_x, rel_y)
    return PointStamped(point=Point(pose.pose.position.x + point.point.x, pose.pose.position.y + point.point.y, 0))

def cal_distance(point_x, point_y):
    dx = abs(point_x.point.x - point_y.point.x)
    dy = abs(point_x.point.y - point_y.point.y)
    return np.hypot(dx, dy)


def get_cone_waypoints(filename):
    cones = []
    df = pd.read_csv(filename)
    longlat_arr = np.array([df['latitude'].to_numpy(),df['longitude'].to_numpy()]).T
    for _, row in df.iterrows():
        waypoint = ConeWaypoints(str(row['id']), row['latitude'], row['longitude'], row['altitude'])
        cones.append(waypoint)

    return cones, longlat_arr

class ConeWaypoints:

    def __init__(self, id, lat, long, alt):
        self.id = id
        self.bbox = []
        self.rel_point = PointStamped()
        self.gps_coord = NavSatFix(latitude=lat, longitude=long, altitude=alt)
        self.tracker = cv2.TrackerKCF_create()
        self.tracked = False

    def reset(self, frame, initBB, get_rel_coord):
        x,y,w,h = initBB[0], initBB[1], initBB[2], initBB[3]
        self.bbox = [x,y,w,h]
        self.rel_point = get_rel_coord(self.bbox, self.id)
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, [int(initBB[0] - 0.5 * initBB[2]),int(initBB[1] - 0.5 * initBB[3]),int(w),int(h)])

    def update(self, frame, get_rel_coord):
        success, box =self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.bbox = [x + 0.5 * w,y + 0.5 * h,w,h]
            updated_rel_point = get_rel_coord(self.bbox, self.id)
            if cal_distance(updated_rel_point, self.rel_point) > 0.5:
                self.tracked = False
                return False
            self.rel_point = updated_rel_point
        self.tracked = success
        return success

if __name__ == '__main__':
    cones, latlong_arr = get_cone_waypoints("data.csv")
    for cone in cones:
        print(cone.gps_location)