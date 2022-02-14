import os
import numpy as np
from tensorflow import keras

def load_model(model_path):

    return keras.models.load_model(model_path)



if __name__ == '__main__':

    model_path = os.getcwd() + '\model\/1'

    model = load_model(model_path)

    #x_test = np.asarray([105.9,105.2,0.7,0.169,2.24,19.2,0.073,0.21,0.142,8.6,8.6,0.558,0.587,0.197,74.98,0.15,2.1,4,0.532,0,0,0,0.7,1,0.729,1.4,4.1,5.6,1.4,1.1,0.1,0.5,0.3,2,1.4,5])
    #print(len(x_test))


    x_test = np.array([[111.2,109.9,1.3,0.128,0.81,9.2,0.072,0.157,0.114,11.4,11.4,0.505,0.565,0.307,80.52,0.137]])
    predictions = model.predict(np.asarray(x_test).astype(np.float32))
    print(predictions)

    # outputs are in type int, convert them to bool
    #y_test = (y_test > 0.5)
    predictions = (predictions > 0.5)
    print(predictions)
