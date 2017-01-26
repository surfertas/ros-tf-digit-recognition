# ROS + Tensorflow + OpenCV 

##Digit Recognition

Digit recognition: trained on MNIST data

Usage:
```
$ roslaunch digit_recognition digit_recognition.launch
```


A first attempt at creating a system that recognizes digits in a scene. The system
uses ROS node system to interlink the jobs, generally broken down by:

1. Capture image
1. Pre-process image
1. Classify image

The associated graph is as follows:

![]()

A couple areas that caused setbacks, were filtering for false hits and freezing
the inference model and reloading for reuse. 
At the moment, the system uses contours to identify possible digits, which results
in an enormous number of possibilities. A simple filter based on geometry is
used to filter out as a first pass, though clearly naive and requires
improvement.

The inference model was trained using a CNN and implemented using the tensorflow
network. The architecture is pretty much off the shelf, with the model trained
using the MNIST data set.

Freezing the model, using steps outline
[here](https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125#.w0ptvk1bd),
was the first step in using an inference model with ROS. The headache came when
reloading the model in a separate node.

There may be a more efficient way of implementing, but at the moment the
solution was to be sensitive to the namespaces of the tensors.

In the digit_classifier node, the tensors related to the graph that was going to
be used needed to be "loaded".

```tf_digit_classifier.py
#model file
self._model = "nodes/tf_model/frozen_model.pb"

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
```
Note: the names for example, "cnn/Placeholder/inputs_placeholder" is completely
user specified.
