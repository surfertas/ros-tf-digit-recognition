#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Camera(object):

    
    def __init__(self):
        rospy.init_node('camera_feed')
        self.image_pub = rospy.Publisher('image_raw', Image, queue_size=1)

    def run(self):
        rate = rospy.Rate(10)

        try:
            while not rospy.is_shutdown():
                #self.image_pub.publish(None)
                rate.sleep()

        except rospy.ROSInterruptException:
            pass    


if __name__=='__main__':
    node = Camera()
    node.run()




