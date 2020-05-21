import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import os
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import numpy as np
from Data import Data

class Triplet(object):
    def __init__(self):
        #----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network
        self.tf_input0 = tf.placeholder(tf.float32, [None, 28,28,1], name = 'input0')
        self.tf_inputPlus = tf.placeholder(tf.float32, [None, 28,28,1], name = 'inputPlus')
        self.tf_inputMinus = tf.placeholder(tf.float32, [None, 28,28,1], name = 'inputMinus')

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name = 'Y')
        self.tf_YOneHot = tf.placeholder(tf.float32, [None,10], name = 'YoneHot')

        # outputs, loss function and training optimizer
        self.output0, self.outputPlus,self.outputMinus = self.tripletNetwork()
        self.output = self.tripletNetworkWithClassification()

        self.loss = self.tripletLoss()
        self.lossCrossEntropy = self.crossEntropyLoss()
        self.optimizer = self.optimizer_initializer()
        self.optimizerCrossEntropy = self.optimizer_initializer_crossEntropy()
        self.saver = tf.train.Saver()

        # Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def layer(self, tf_input, num_hidden_units, variable_name, trainable=True):
        # tf_input: batch_size x n_features
        # num_hidden_units: number of hidden units
        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)
        num_features = tf_input.get_shape()[1]
        W = tf.get_variable(
            name = variable_name + '_W',
            dtype = tf.float32,
            shape = [num_features, num_hidden_units],
            initializer = tf_weight_initializer,
            trainable=trainable
            )
        b = tf.get_variable(
            name = variable_name + '_b',
            dtype = tf.float32,
            shape = [num_hidden_units],
            trainable=trainable
            )
        out = tf.add(tf.matmul(tf_input, W), b)
        return out

    def maxpool2d(self,x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    def network(self, tf_input, trainable=True):
        # Setup CNN
        with tf.variable_scope("triplet") as scope:
            fc1= tf.layers.Conv2D(filters=4,kernel_size=(3,3),strides=1,padding='VALID',trainable=trainable)(tf_input)
            #ac1 = tf.nn.relu(fc1)
            ac1=tf.nn.tanh(fc1)
            ac1_pooled=self.maxpool2d(ac1)
            fc2=tf.layers.Conv2D(filters=8,kernel_size=(3,3),strides=1,padding='VALID',trainable=trainable)(ac1_pooled)
            #ac2 = tf.nn.relu(fc2)
            ac2=tf.nn.tanh(fc2)
            ac2_pooled=self.maxpool2d(ac2)
            out=tf.layers.Flatten()(ac2_pooled)
            print(out.get_shape())#(?,28*23*8)=(?,5152)
            return out #embeddings that have the most relative info helpful for further classification

    def networkWithClassification(self, tf_input):
        # Setup CNN
        fc3 = self.network(tf_input, trainable=False)
        ac3 = tf.nn.relu(fc3)
        fc4 = self.layer(tf_input = ac3, num_hidden_units = 200, trainable=True, variable_name = 'fc4')
        ac4 = tf.nn.relu(fc4)
        fc5 = self.layer(tf_input = ac4, num_hidden_units = 10, trainable=True, variable_name = 'fc5')
        return fc5

    def tripletNetwork(self):
        # Initialze neural network
        with tf.variable_scope("triplet") as scope:
            output0 = self.network(self.tf_input0)
            # share weights
            scope.reuse_variables()
            outputPlus = self.network(self.tf_inputPlus)
            scope.reuse_variables()
            outputMinus = self.network(self.tf_inputMinus)
        return output0, outputPlus, outputMinus

    def tripletNetworkWithClassification(self):
        # Initialze neural network
        with tf.variable_scope("triplet",reuse=tf.AUTO_REUSE) as scope:
            output = self.networkWithClassification(self.tf_input0)
        return output

    def contastiveLoss(self, margin = 5.0):# not used in triplet
        with tf.variable_scope("triplet") as scope:
            labels = self.tf_Y
            # Euclidean distance squared
            dist = tf.pow(tf.subtract(self.outputA, self.outputB), 2, name = 'Dw')
            Dw = tf.reduce_sum(dist, 1)
            # add 1e-6 to increase the stability of calculating the gradients
            Dw2 = tf.sqrt(Dw + 1e-6, name = 'Dw2')
            # Loss function
            lossSimilar = tf.multiply(labels, tf.pow(Dw2,2), name = 'constrastiveLoss_1')
            lossDissimilar = tf.multiply(tf.subtract(1.0, labels), tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2), name = 'constrastiveLoss_2')
            loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name = 'constrastiveLoss')
        return loss

    def tripletLoss(self):
        with tf.variable_scope("triplet") as scope:
            negative_squared = tf.pow(tf.subtract(self.output0, self.outputPlus), 2)
            magnitude_negative = tf.sqrt(tf.reduce_sum(negative_squared, 1))

            positive_squared = tf.pow(tf.subtract(self.output0, self.outputMinus), 2)
            magnitude_positive = tf.sqrt(tf.reduce_sum(positive_squared, 1))

            e_to_positive  = tf.exp(magnitude_positive)
            e_to_negative = tf.exp(magnitude_negative)
            denom_softmax = tf.add(e_to_positive, e_to_negative)
            #Softmax
            d_plus = tf.math.divide(e_to_positive, denom_softmax)
            d_minus = tf.math.divide(e_to_negative, denom_softmax)

            loss = tf.pow(d_plus, 2) + tf.pow(tf.subtract(d_minus, 1), 2)
            loss = tf.reduce_mean(loss)
            return loss
            '''dist=[]
            dist.append(self.euclideanDist(self.output0,self.outputPlus))#dist[0]
            dist.append(self.euclideanDist(self.output0,self.outputMinus))#dist[1]
            softmax_d=tf.nn.softmax(dist)
            loss=tf.reduce_mean(tf.pow(tf.subtract(dist,[0,1]), 2))
        return loss'''

    
    def euclideanDist(self,x1,x2):
        dist = tf.reduce_sum(tf.sqrt(tf.pow(tf.subtract(x1, x2), 2)))
        return dist


    def crossEntropyLoss(self):
        labels = self.tf_YOneHot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=labels))
        return lossd

    def optimizer_initializer(self):
        LEARNING_RATE = 0.01
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)
        return optimizer

    def optimizer_initializer_crossEntropy(self):
        LEARNING_RATE = 0.001
        RAND_SEED = 0 # random seed
        tf.set_random_seed(RAND_SEED)
        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.lossCrossEntropy)
        return optimizer

    def trainTriplet(self, x, y, epochs, batchSize=10):#change for 3 inputs/outputs
        # Train the network
        data = Data(x,y)
        #numBatches=numSamples//batchSize
        #numS
        #amples=batchSize*numBatches
        for i in range(epochs):
            data.assemple_in_triplets()
            numBatches=len(data.list_of_triplets)//batchSize
            numSamples=batchSize*numBatches
            for j in range(0,numSamples,batchSize):
                X = data.get_nextBatch(batchSize)
                x0=X[:,0,:,:]
                xPlus=X[:,1,:,:]
                xMinus=X[:,2,:,:]
                input2 = x[j+batchSize:j+2*batchSize]
                _, trainingLoss = self.sess.run([self.optimizer, self.loss],
                feed_dict = {self.tf_input0: x0, self.tf_inputPlus: xPlus, self.tf_inputMinus: xMinus})
                #print(trainingLoss)
            print('iteration %d: train loss %.3f' % (i, trainingLoss))

    def trainTripletForClassification(self, x, y, epochs, batchSize=10):
        # Train the network for classification via softmax
        numSamples=x.shape[0]
        numBatches=numSamples//batchSize
        numSamples=batchSize*numBatches
        for i in range(epochs):
            x,y=shuffle(x,y)
            for j in range(0,numSamples,2*batchSize):
                input1 = x[j:j+batchSize]
                y1=y[j:j+batchSize]
                y1c = to_categorical(y1,num_classes=10) # convert labels to one hot
                labels = np.zeros(batchSize)
                _, trainingLoss = self.sess.run([self.optimizerCrossEntropy, self.lossCrossEntropy],
                feed_dict = {self.tf_input0: input1, self.tf_inputPlus: input1,self.tf_inputMinus: input1,self.tf_YOneHot: y1c, self.tf_Y:labels})
            print('iteration %d: train loss %.3f' % (i, trainingLoss))


    def computeAccuracy(self,x,y):
        labels = np.zeros(100)
        yonehot = np.zeros((100,10))
        aout = self.sess.run(self.output, feed_dict={self.tf_input0: x,
        self.tf_inputPlus: x,self.tf_inputMinus: x, self.tf_YOneHot: yonehot, self.tf_Y:labels})
        accuracyCount = 0
        testY = to_categorical(y) # one hot labels
        for i in range(testY.shape[0]):
            # determine index of maximum output value
            maxindex = aout[i].argmax(axis = 0)
            if (testY[i,maxindex] == 1):
                accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')
        
    def test_model(self, input): #???change for 3 inputs/outputs
        # Test the trained model
        output = self.sess.run(self.output0, feed_dict = {self.tf_input0: input})
        return output 

