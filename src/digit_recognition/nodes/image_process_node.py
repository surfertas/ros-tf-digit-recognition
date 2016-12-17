#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

class ImageProcessor(object):

    
    def __init__(self):
        self._BRIDGE = CvBridge()

        rospy.init_node('image_processor')
        self._image_sub = rospy.Subscriber('/image_raw', Image, self._image_process_cb)
        self._image_pub = rospy.Publisher('image_processed', Image, queue_size=1)

    def _add_padding(self, x, y, w, h, shape):
        """ Adds padding around the roi.

        Args:
            x:  x-axis coordinate
            y:  y-axis coordinate
            w:  width of roi
            h:  heigth of roi
            shape: shape of image, used for boundaries

        Returs:
            x1,x2,y1,y2:    coordinates of padded roi
            
        """
        x_max, y_max, _ = shape

        x1 = x * 0.8
        x2 = min(int((x + w) * 1.2), x_max)
        y1 = y * 0.8
        y2 = min(int((y + h) * 1.2), y_max)
        
        return x1, x2, y1, y2

    def _image_process_cb(self, image):
        try:
            cv_img = self._BRIDGE.imgmsg_to_cv2(image, "8UC3")

        except CvBridgeError as e:
            print(e)

        #convert to gray image, and blur
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) 
        blur_img = cv2.GaussianBlur(gray_img, (3,3), 0)


        #create binary image
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 
                                31,10)

        thresh_img = cv2.medianBlur(thresh_img, 3)

        #find contours
        img, ctrs, _ = cv2.findContours(thresh_img.copy(), 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)

        #bounding rectangles with condition to filter false hits
        bounded = [cv2.boundingRect(ctr) for ctr in ctrs
                   if cv2.contourArea(ctr) > 200]


        for rect in bounded:

            x,y,w,h = rect

            #filter for false hits
            if h/float(w) > 0.5:
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            x1,x2,y1,y2 = self._add_padding(x, y, w, h, cv_img.shape)

            roi = thresh_img[y1:y2, x1:x2]

            roi = cv2.resize(roi,(28,28)) / 255.

            img_msg = self._BRIDGE.cv2_to_imgmsg(roi, encoding="passthrough")
            self._image_pub.publish(img_msg)

            
        #cv2.imshow("Image", cv_img)
        #cv2.waitKey(3)


    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()


if __name__=='__main__':
    node = ImageProcessor()
    node.run()
