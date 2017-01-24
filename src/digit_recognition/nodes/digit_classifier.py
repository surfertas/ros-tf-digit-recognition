#!/usr/bin/env python 
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sklearn import neural_network
from sklearn.externals import joblib
from keras.models import model_from_json
import numpy as np

from keras import backend as K
from theano import function

from generate_cnn import MLP

MODEL_FILE_PATH = "digit_clf.cnn.sav"

class DigitClassifier(object):
    

    def __init__(self):                
        self._BRIDGE = CvBridge()

        rospy.init_node('digit_classifier')
        self._image_sub = rospy.Subscriber('/image_processed', Image, self._classify_cb)
        
        self._display = True
        self._count = 0 
        self._model = self._load_model()


    def _load_model(self):
        model = joblib.load(MODEL_FILE_PATH)

        json_file = open('digit_clf_cnn.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("digit_clf_cnn_w.h5")

        model._model.compile(loss='binary_crossentropy', 
                             optimizer='rmsprop', 
                             metrics=['accuracy'])
        return model


    def _classify_cb(self, image):
        self._count += 1
        if self._count % 50 == 0: 
            self._display = True

        if self._display:
            try:
                cv_img = self._BRIDGE.imgmsg_to_cv2(image)#, "8UC3")

            except CvBridgeError as e:
                print(e)

            
            #un-hash to see image
            cv2.imshow("Image", cv_img)
            cv2.waitKey(3)

            #img = np.reshape(cv_img, (-1,784))
            img = np.reshape(cv_img, (-1, 28, 28, 1)).astype('float32')

            digit_arr = self._model.predict_proba(img)
            i = np.argmax(digit_arr[0])
            
            if self._display:
                if digit_arr[0,i] > 0.9:
                    print(i, digit_arr[0,i])
                
                self._display = False

            

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()
    
if __name__=='__main__':
    clf = DigitClassifier()
    clf.run()


