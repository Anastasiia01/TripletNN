#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from Triplet import Triplet
import numpy as np
#from tensorflow.keras.utils import to_categorical

def visualize(embed, labels):
    labelset = set(labels)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
    ax.legend()
    #fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

    #s  = pd.Series(y_test)
    #y_test=pd.get_dummies(s)

   
    triplet = Triplet()
    triplet.trainTriplet(x_train, y_train, 7, 20)
    triplet.trainTripletForClassification(x_train, y_train, 10, 20)
    
    # Test model
    embed = triplet.test_model(input = x_test)
    #embed.tofile('embed.txt')
    #embed = np.fromfile('embed.txt', dtype = np.float32)

    #embed = embed.reshape([-1, 2])
    #visualize(embed, y_test)

    triplet.computeAccuracy(x_test,y_test)


if __name__ == '__main__':
    main()




