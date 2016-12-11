import sys

from sklearn.externals import joblib 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils

import numpy as np 

sys.setrecursionlimit(10000)

SEED = 10
np.random.seed(SEED)

class MLP(object):
    

    def __init__(self, num_inputs, num_labels):
        self._num_inputs = num_inputs
        self._num_labels = num_labels
        self._model = self._initialize_model()

    def __call__(self, X, Y):
        return self._model.evaluate(X, Y, verbose=0)

    def _initialize_model(self):
        """Defines multilayer neral network
        """
        model = Sequential()
        model.add(Dense(self._num_inputs, input_dim=self._num_inputs, init='normal'))
        model.add(Activation('relu'))
        model.add(Dense(self._num_labels, init='normal'))
        model.add(Activation('softmax'))
                
        print("Compiling model...")
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        return model
    
    def process_data(self, X, Y):        
        """Splits original data set into training and test data.
    
        Args:
            X: Raw data set of features.
            Y: Raw data set of labels.

        Returns:
            List: [X_train, X_test, y_train, y_test]
        
        """
        return train_test_split(X, Y, test_size=0.33, random_state=42)

    def train(self, X, Y):
        """Preprocess data then fits to model. Validation process
        included in method.

        Args:
            X: Raw data set of features.
            Y: Raw data set of labels.
        
        """
        X_train, X_test, y_train, y_test = self.process_data(X,Y)    

        self._model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        nb_epoch=10, 
                        batch_size=200,
                        verbose=2)

        print("Model trained...")

if __name__=="__main__":

    #pulls raw data
    dataset = datasets.fetch_mldata("MNIST Original")
    
    #normalize by 255
    X = np.array(dataset.data, 'float32')  / 255. 
    
    #one hot encode
    Y = np_utils.to_categorical(np.array(dataset.target, 'float32'))

    #data dimensions
    nfeatures = X.shape[1]
    nlabels = Y.shape[1]
   
    model = MLP(nfeatures, nlabels)
    X_train, X_test, y_train, y_test = model.process_data(X,Y)    

    model.train(X_train,y_train)
    results = model(X_test, y_test)
    print("Accuracy: %.2f%%" % (results[1]*100))

    #save model
    model_name = "digit_classifier.sav"
    joblib.dump(model, open(model_name, 'wb'))
    

