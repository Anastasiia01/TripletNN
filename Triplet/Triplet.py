import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import os
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import numpy as np

class Triplet(object):
    def __init__(self):
        #----set up place holders for inputs and labels for the siamese network---
        # two input placeholders for Siamese network
        self.tf_inputA = tf.placeholder(tf.float32, [None, 28,28,1], name = 'inputA')
        self.tf_inputB = tf.placeholder(tf.float32, [None, 28,28,1], name = 'inputB')

        # labels for the image pair # 1: similar, 0: dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None,], name = 'Y')
        self.tf_YOneHot = tf.placeholder(tf.float32, [None,10], name = 'YoneHot')

        # outputs, loss function and training optimizer
        self.outputA, self.outputB = self.tripletNetwork()
        self.output = self.tripletNetworkWithClassification()

        self.loss = self.contastiveLoss()
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
            fc1= tf.layers.Conv2D(filters=4,kernel_size=(3,3),strides=1,padding='SAME',trainable=trainable)(tf_input)
            #fc1 = self.cnnLayer(tf_input = tf_input, feature_maps_out=4, kSize=3,trainable=trainable, variable_name = 'fc1')
            #ac1 = tf.nn.relu(fc1)
            ac1=tf.nn.tanh(fc1)
            ac1_pooled=self.maxpool2d(ac1)
            fc2=tf.layers.Conv2D(filters=8,kernel_size=(3,3),strides=1,padding='SAME',trainable=trainable)(ac1_pooled)
            #ac2 = tf.nn.relu(fc2)
            ac2=tf.nn.tanh(fc2)
            ac2_pooled=self.maxpool2d(ac2)
            out=tf.layers.Flatten()(ac2_pooled)
            print(out.get_shape())#(?,28*23*8)=(?,5152)
            return out #embeddings that have the most relative info helpful for further classification

    def networkWithClassification(self, tf_input):
        # Setup FNN
        fc3 = self.network(tf_input, trainable=False)
        ac3 = tf.nn.relu(fc3)
        fc4 = self.layer(tf_input = ac3, num_hidden_units = 200, trainable=True, variable_name = 'fc4')
        ac4 = tf.nn.relu(fc4)
        fc5 = self.layer(tf_input = ac4, num_hidden_units = 10, trainable=True, variable_name = 'fc5')
        return fc5

    def tripletNetwork(self):
        # Initialze neural network
        with tf.variable_scope("triplet") as scope:
            outputA = self.network(self.tf_inputA)
            # share weights
            scope.reuse_variables()
            outputB = self.network(self.tf_inputB)
        return outputA, outputB

    def tripletNetworkWithClassification(self):
        # Initialze neural network
        with tf.variable_scope("triplet",reuse=tf.AUTO_REUSE) as scope:
            output = self.networkWithClassification(self.tf_inputA)
        return output

    def contastiveLoss(self, margin = 5.0):
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

    def trainTriplet(self, x, y, epochs, batchSize=10):
        # Train the network
        numSamples=x.shape[0]
        numBatches=numSamples//batchSize
        numSamples=batchSize*numBatches
        for i in range(epochs):
            x,y=shuffle(x,y)
            for j in range(0,numSamples,2*batchSize):
                input1 = x[j:j+batchSize]
                y1=y[j:j+batchSize]
                input2 = x[j+batchSize:j+2*batchSize]
                y2=y[j+batchSize:j+2*batchSize]
                label = (y1 == y2).astype('float')
                _, trainingLoss = self.sess.run([self.optimizer, self.loss],
                feed_dict = {self.tf_inputA: input1, self.tf_inputB: input2, self.tf_Y: label})
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
                feed_dict = {self.tf_inputA: input1, self.tf_inputB: input1,self.tf_YOneHot: y1c, self.tf_Y:labels})
            print('iteration %d: train loss %.3f' % (i, trainingLoss))


    def computeAccuracy(self,x,y):
        labels = np.zeros(100)
        yonehot = np.zeros((100,10))
        aout = self.sess.run(self.output, feed_dict={self.tf_inputA: x,
        self.tf_inputB: x,self.tf_YOneHot: yonehot, self.tf_Y:labels})
        accuracyCount = 0
        testY = to_categorical(y) # one hot labels
        for i in range(testY.shape[0]):
            # determine index of maximum output value
            maxindex = aout[i].argmax(axis = 0)
            if (testY[i,maxindex] == 1):
                accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')
        
    def test_model(self, input):
        # Test the trained model
        output = self.sess.run(self.outputA, feed_dict = {self.tf_inputA: input})
        return output 

