from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import os
import numpy as np
import itertools

class Data(object):

    def __init__(self, x_train, y_train):
        self.x=x_train
        self.y=y_train
        self.list_of_triplets=[]#list of tuples of three
        self.my_iter=None        

    def assemple_in_triplets(self):
        self.x,self.y=shuffle(self.x,self.y)
        self.list_of_triplets=[]
        for i in range(len(self.x)):
            input0=self.x[i]
            inputPlus=None
            inputMinus=None
            needed=1
            for j in range(i+1,len(self.x)):
                if(needed==1):
                    if(self.y[i]==self.y[j]):
                        inputPlus=self.x[j]
                        needed=2
                elif(needed==2):  
                    if(self.y[i]!=self.y[j]):
                        inputMinus=self.x[j]
                        #print((input0,inputPlus,inputMinus))
                        self.list_of_triplets.append([input0,inputPlus,inputMinus])
                        needed=0        
                else:#needed==0
                    break
        #print(len(self.list_of_triplets))
        self.my_iter=iter(self.list_of_triplets)


    def get_nextBatch(self, batchSize):#use batch instead of singular
        try:
            res=[]
            for i in range(batchSize):
                res.append(next(self.my_iter))
            if(len(res)==batchSize):
                return np.array(res)
        except:
            return []



