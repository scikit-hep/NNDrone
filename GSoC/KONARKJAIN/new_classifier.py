from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pickle
from utilities.plotting import hd_hist, scatter
import matplotlib.pyplot as plt
from utilities.utilities import next_batch
import matplotlib.pyplot as plt

import sys
import os, errno

class Network:
    def __init__(self, Topo, Train, Test):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
        self.lrate = 0.05
        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)
    def evaluate_proposal(self, data):  # BP with SGD (Stocastic BP)

        size = data.shape[0]
        w = 0
        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros((size,1))

        for pat in range(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]
            self.ForwardPass(Input)
            if self.out > 0.5:
                fx[pat] = 1
            self.BackwardPass(Input, Desired)
        #print(fx.shape)
        return [fx, w]
#DATA LOADING
sig_data = joblib.load('signal_data.p') 
bkg_data = joblib.load('background_data.p')
sig_X=np.asarray(sig_data)
bkg_X=np.asarray(bkg_data)
sig_Y=np.ones((sig_X.shape[0],1))
bkg_Y=np.zeros((bkg_X.shape[0],1))
data=np.concatenate((sig_X,bkg_X),axis=0)        
label=np.concatenate((sig_Y,bkg_Y),axis=0)
perm=np.random.permutation(data.shape[0])
data=data[perm,:]
label=label[perm,:]
#
#Feature select
a = 0
b = 2
data = data[:,[a,b]]
#
trainFraction = 0.5 #can be chosen
cutIndex = int(trainFraction * len(label))
#
dataTrain = data[: cutIndex]
labelTrain = label[: cutIndex]
dataTest = data[cutIndex:]
labelTest = label[cutIndex:]
#MODEL
topology = [2,10,1]
net = Network(topology,np.hstack([dataTrain,labelTrain]),np.hstack([dataTest,labelTest]))
loss = []
for epoch in range(5000):
    [fx, w] = net.evaluate_proposal(net.TrainData)
    diff = fx - labelTrain
    #print fx.shape
    loss = np.append(loss,diff.T.dot(diff))
    if epoch%200 == 0:
        print("Cost after epoch {} {}".format(epoch,loss[epoch]))
        
fig = plt.figure()
plt.plot(range(5000),loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")
plt.clf()

train_accuracy = 1 - (np.sum(np.absolute(diff))/data.shape[0])
print "Train accuracy={}".format(train_accuracy)

[fx, w] = net.evaluate_proposal(net.TestData)
diff = fx - labelTest
test_accuracy = 1 - (np.sum(np.absolute(diff))/data.shape[0])
print "Test accuracy={}".format(test_accuracy) 

    

