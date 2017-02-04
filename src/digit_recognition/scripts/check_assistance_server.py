#!/usr/bin/env python

#TODO:
#http://wiki.ros.org/rospy_tutorials/Tutorials/Makefile
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from digit_recognition.srv import *


class DigitCheckService(object):

    def __init__(self):
        self._BRIDGE = CvBridge()
        rospy.init_node('assistance_service')
        service = rospy.Service('assistance_service', DigitCheck, self._handle_assistance)

    def _handle_assistance(self, request):
        try:
            cv_img = self._BRIDGE.imgmsg_to_cv2(request.img)
        except CvBridgeError as e:
            return

        cv2.imshow("Image", cv_img)
        cv2.waitKey(10)
        while True:
            try:
                ans = int(raw_input('Yes: 1, No: 0\n'))
            except ValueError:
                print("Need a 1 or 0\n")

            if ans in [0,1]:
                break
        
        return DigitCheckResponse(ans)
        

    def run_assistance(self):
        try:     
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()


if __name__=="__main__":
    service = DigitCheckService()
    service.run_assistance() 
