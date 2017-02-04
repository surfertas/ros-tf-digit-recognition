#!/usr/bin/env python 
import rospy
import rospkg
import tensorflow as tf
import numpy as np
import cv2
import os

from sensor_msgs.msg import Image
from digit_recognition.srv import *
from cv_bridge import CvBridge, CvBridgeError

MODEL_FILE_PATH = "tf_cnn_model.ckpt.meta"
WEIGHTS_FILE_PATH = "tf_cnn_model.ckpt"


class DigitClassifier(object):
    

    def __init__(self):                
        self._BRIDGE = CvBridge()

        rospy.init_node('digit_classifier')
        rospy.wait_for_service('assistance_service')
        self._digit_check = rospy.ServiceProxy('assistance_service', DigitCheck)
        self._rospack = rospkg.RosPack()

        self._image_sub = rospy.Subscriber('/image_processed', Image, self._classify_cb)
        
        #model file
        self._model = "scripts/tf_model/frozen_model.pb"

        #set tf graph
        self._graph = self._load_graph(self._model)
        
        #tf related tensors
        self.x = self._graph.get_tensor_by_name('cnn/Placeholder/inputs_placeholder:0')
        self.y_ = self._graph.get_tensor_by_name('cnn/Placeholder/labels_placeholder:0')
        self.guess = self._graph.get_tensor_by_name('cnn/Accuracy/ArgMax:0')
        self.output = self._graph.get_tensor_by_name('cnn/NN/output:0')
        self.keep_prob = self._graph.get_tensor_by_name('cnn/Placeholder/keep_prob:0')
        self.accuracy = self._graph.get_tensor_by_name('cnn/Accuracy/accuracy:0')

        #start session
        self._sess = tf.Session(graph=self._graph)

    def _load_graph(self, frozen_graph_filename):
        
        ros_path = self._rospack.get_path('digit_recognition')
        path = os.path.join(ros_path, frozen_graph_filename)

        with tf.gfile.GFile(path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="cnn",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def _classify_cb(self, image):
        if self._digit_check(image):
            try:
                cv_img = self._BRIDGE.imgmsg_to_cv2(image)

            except CvBridgeError as e:
                print(e)


            img = np.reshape(cv_img, (-1, 28, 28, 1)).astype('float32')
            img[img<0.05] = 0

            predict = self._sess.run(self.output, 
                                     feed_dict={self.x: img, self.keep_prob:1.0})
                  
            print(np.argmax(predict,1)[0])

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()

    
if __name__=='__main__':
    clf = DigitClassifier()
    clf.run()


