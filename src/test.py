#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
import time

from nav_msgs.msg import Odometry

from bondpy import bondpy

import numpy as np

x_data = []
y_data = []


def listener(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    x_data.append(x)
    y_data.append(y)
    

rospy.init_node('listener')

sub = rospy.Subscriber("/mavros/global_position/local", Odometry, callback=listener, queue_size=1)


while not rospy.is_shutdown():
    plt.plot(x_data, y_data, color="r")
    plt.draw()
    plt.pause(0.01)

plt.show()
