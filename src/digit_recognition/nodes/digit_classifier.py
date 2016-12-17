#!/usr/bin/env python 
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sklearn import neural_network
from sklearn.externals import joblib
import numpy as np

from generate_model import MLP

MODEL_FILE_PATH = "digit_classifier.sav"

class DigitClassifier(object):
    

    def __init__(self):                
        self._BRIDGE = CvBridge()

        rospy.init_node('digit_classifier')
        self._image_sub = rospy.Subscriber('/image_processed', Image, self._classify_cb)
        
        self._model = joblib.load(MODEL_FILE_PATH)

    def _classify_cb(self, image):

        try:
            cv_img = self._BRIDGE.imgmsg_to_cv2(image)#, "8UC3")

        except CvBridgeError as e:
            print(e)

        
        #un-hash to see image
        cv2.imshow("Image", cv_img)
        cv2.waitKey(3)

        img = np.reshape(cv_img, (1,784))

        #TODO: Only output high probability of digot
        print(np.argmax(self._model.predict(img)))

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()
    
if __name__=='__main__':
    clf = DigitClassifier()
    clf.run()


