#!/usr/bin/python3
"""
This file uses the rgb and depth feed from the topics published by a Intel RealSense camera to detect people and
returns their poses relative to the frame of the camera.
Uses the jetson-inference package found here: https://github.com/dusty-nv/jetson-inference
"""
import rospy
import math
import message_filters

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ppl_detect import PeopleDetection

def map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# function that is called every time there is a new image frame that the ROS subscriber receives
def callback(rgb_image, depth_image, pd_class, bridge, detection_pub, pub_command):
    """
    callback function that uses rgb frame and depth frame to detect people and prints out their (x,y,z) coordinate
    from the perspective of the camera
    :param rgb_image: rgb frame from realsense camera
    :param depth_image: depth frame which is mapped to same timestamp as rgb frame from realsense camera
    :param pd_class: object instantiated of class PeopleDetection()
    :param bridge: openCV bridge to convert frame to numpy array and vice-versa
    :param detection_pub: rospy publisher node used to publish resulting image with bounding boxes, labels,
    and confidence percentage
    """
    bridge = bridge
    ppl_detect_class = pd_class
    cv_rgb = bridge.imgmsg_to_cv2(rgb_image, "rgba8")
    cv_depth = bridge.imgmsg_to_cv2(depth_image, "passthrough")
    detections, result_img = ppl_detect_class.get_detections(cv_rgb)
    detection_pub.publish(bridge.cv2_to_imgmsg(result_img, "rgba8"))
    #print("detected {:d} objects in image".format(detections.n))
    coord_results = ppl_detect_class.get_person_coordinates(cv_depth, detections)
    if coord_results:
        cmd = get_controls(coord_results[0])
        pub_command.publish(cmd)
    #print(coord_results)

def get_controls(target_coord):
    twist = Twist()
    x,y = target_coord[0], target_coord[2]
    angular_z = math.atan2(-x,y)
    linear_x = map(y, 1, 5, 0, 1)
    #print(f"{angular_z:.2f} {linear_x:.2f}")
    twist.linear.x = linear_x
    twist.angular.z = angular_z
    return twist


def main():
    """
    Initializes rospy subscriber and publisher nodes, object of PeopleDetection class
    Synchronizes rgb and depth frames for matching frames
    Runs indefinitely until user stops
    """
    rospy.init_node('PeopleDetection', anonymous=True)
    print("Running People Detection")

    people_detect = PeopleDetection()
    bridge = CvBridge()
    pub_command = rospy.Publisher('/robot/navigation/input', Twist, queue_size=1)
    detection_pub = rospy.Publisher("detected_image", Image, queue_size=1)
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1)
    depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 5, allow_headerless=True)
    ts.registerCallback(callback, people_detect, bridge, detection_pub, pub_command)
    rospy.spin()


if __name__ == '__main__':
    main()
