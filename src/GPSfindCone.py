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
#	Utility Functions used in controlNode.py to estimate the position of the cone based on the depth camera
#   and the vehicle GPS data
########################################

import numpy as np
from geographiclib.geodesic import Geodesic


def get_distance_and_bearing(lat1, long1, lat2, long2):
    ######################################
    '''
    Uses the geographic library to calculate the distance and bearing between two points

    Input: lat1: latitude of first point,
            long1: longitude of first point, 
            lat2: latitude of second point,
            long2: longitude of second point

    Output: distance: distance between two points,
            bearing: bearing of point 2 relative to point 1
        
    '''
    ######################################
    geodict = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)
    bearing = geodict['azi1'] 
    if bearing < 0: bearing += 360
    return geodict['s12'], bearing

def get_cone_coord(car_lat, car_long, camera_x, camera_y, heading):
    ######################################
    '''

    Returns the estimated GPS coordinate of the cone based on the car's GPS data and the depth camera image

    Input:  car_lat: latitude of car,
            car_long: longitude of car, 
            camera_x: x-axis position of cone relative to the camera,
            camera_y: y-axis position of cone relative to the camera,
            heading: current compass heading of the car

    Output: latitude and longitude of the car
            
    '''
    ######################################
    theta = np.deg2rad(heading)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    world_pos = np.matmul(np.linalg.inv(R), np.array([[camera_x, camera_y]]).T)
    #print(world_pos)
    dist = np.linalg.norm(world_pos,axis=0)
    angle = -np.rad2deg(np.arctan2(world_pos[1,0], world_pos[0,0]))+90
    geodict = Geodesic.WGS84.Direct(car_lat, car_long, angle, dist)
    
    return geodict['lat2'], geodict['lon2']

if __name__ == '__main__':

    # tests to verify that the code is working 
    lat1, long1, lat2, long2 = 47.673057, -122.310880, 47.673274, -122.211058
    dist, bearing = get_distance_and_bearing(lat1, long1, lat2, long2)
    print("distance: " + str(dist) + " bearing: " + str(bearing))

    camera_x, camera_y = 0, 5
    
    heading = 0

    coord = get_cone_coord(lat1, long1, 0, 5, heading)
    print(coord)