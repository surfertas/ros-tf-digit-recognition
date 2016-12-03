#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor(object):

    
    def __init__(self):
        self._BRIDGE = CvBridge()

        rospy.init_node('image_processor')
        self._image_sub = rospy.Subscriber('/image_raw', Image, self._image_process_cb)

    def _image_process_cb(self, image):
        try:
            cv_image = self._BRIDGE.imgmsg_to_cv2(image, "8UC3")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image", cv_image)
        cv2.waitKey(3)

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()


if __name__=='__main__':
    node = ImageProcessor()
    node.run()
