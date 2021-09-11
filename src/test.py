import numpy as np
from geographiclib.geodesic import Geodesic

def get_distance_and_bearing(lat1, long1, lat2, long2):
    geodict = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)
    bearing = geodict['azi1'] 
    if bearing < 0: bearing += 360
    return geodict['s12'], bearing

def get_cone_coord(car_lat, car_long, camera_x, camera_y, heading):
    theta = np.deg2rad(heading)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    world_pos = np.matmul(np.linalg.inv(R), np.array([[camera_x, camera_y]]).T)
    print(world_pos)
    dist = np.linalg.norm(world_pos,axis=0)
    angle = -np.rad2deg(np.arctan2(world_pos[1,0], world_pos[0,0]))+90
    geodict = Geodesic.WGS84.Direct(car_lat, car_long, angle, dist)
    
    return geodict['lat2'], geodict['lon2']

if __name__ == '__main__':
    lat1, long1, lat2, long2 = 47.673057, -122.310880, 47.672974, -122.311058
    dist, bearing = get_distance_and_bearing(lat1, long1, lat2, long2)
    print("distance: " + str(dist) + " bearing: " + str(bearing))

    camera_x, camera_y = 0, 5
    
    heading = 0

    coord = get_cone_coord(lat1, long1, 0, 5, heading)
    print(coord)