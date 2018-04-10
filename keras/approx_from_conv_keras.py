#!/usr/bin/python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:44:42 2017

@author: sbenson
@author: kgizdov

# Demonstrator that model conversion works
"""

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

from keras.layers import Conv1D, LocallyConnected1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import load_model, Sequential

try:
    from plotting import hd_hist, scatter
except ImportError:
    from utilities.plotting import hd_hist, scatter

try:
    from utilities import dot_loss, next_batch
except ImportError:
    from utilities.utilities import dot_loss, next_batch

try:
    from models import BaseModel as Model
except ImportError:
    from models.models import BaseModel as Model

try:
    from converters import BasicConverter
except ImportError:
    from utilities.converters import BasicConverter

# DATASET LOADING ***************************************************************
totalDataSig = []
totalDataBkg = []
totalData = []
probabilitiesSig = []
probabilitiesBkg = []

print('Loading classifier...')
classifier = load_model("./keras_locallyconnected1d_for_drone.h5")

print('Loading signal data file...')
# sig_data = joblib.load('../data/signal_data_gpd.p')
sig_data = joblib.load('../data/signal_data.p')
print('Loading background data file...')
# bkg_data = joblib.load('../data/background_data_gpd.p')
bkg_data = joblib.load('../data/background_data.p')
#
trainFraction = 0.5
cutIndex = int(trainFraction * len(sig_data))
#
sigTrain = sig_data[: cutIndex]
sigTest = sig_data[cutIndex:]
#
bgTrain = bkg_data[: cutIndex]
bgTest = bkg_data[cutIndex:]

# Create the scaler to preprocess the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(sigTrain)

# transform the training sameple
sig_train = scaler.transform(sigTrain)
# do the same to the test data
sig_test = scaler.transform(sigTest)
# do the same to the test data
bkg_train = scaler.transform(bgTrain)
# do the same to the test data
bkg_test = scaler.transform(bgTest)


def fname(prefix, base, a, n, t, suff):
    return '%s%s_alpha%s_epochs%s_thresh%s.%s' % (prefix, base, a, n, t, suff)

for s in sig_train:
    s = scaler.transform([s])
    # prob = classifier.predict_proba(b)[0][0]
    prob = classifier.predict_proba(np.expand_dims(s, axis = 2))[0][0]
    s = s[0].flatten().tolist()
    probabilitiesSig.append(prob)
    totalDataSig.append(s)
for b in bkg_train:
    b = scaler.transform([b])
    # prob = classifier.predict_proba(b)[0][0]
    prob = classifier.predict_proba(np.expand_dims(np.expand_dims(s, axis = 2), axis = 0))[0][0]
    b = b[0].flatten().tolist()
    probabilitiesBkg.append(prob)
    totalDataBkg.append(b)

# initialize a list to store the loss value for each epoch
lossHistory = []

# Approx. control
alpha = 0.05
batchSize = 1
# num_epochs = 1500
num_epochs = 300
threshold = 0.02

# layerSizes = [300, 1]
layerSizes = [5, 1]


train = totalDataSig+totalDataBkg
# train = totalDataSig
datasize = len(train)
probs_train = probabilitiesSig+probabilitiesBkg
# probs_train = probabilitiesSig

# convert features and outputs to np array
train = np.array(train)
probs_train = np.array(probs_train)

# shuffle signal and background
s = np.arange(train.shape[0])
np.random.shuffle(s)
train = train[s]
probs_train = probs_train[s]

# Initialise model
model = Model(len(train[0]), 1)
if len(layerSizes) == 1:
    model.add_layer(1)
else:
    for l in layerSizes:
        model.add_layer(l)

model.print_layers()

print("Starting stochastic conversion...")
# loop over the desired number of epochs
updatedLoss = 1000.0
diffs = []
losses = []
updates = []
for q in range(num_epochs):
    # initialize the total loss for the epoch
    epochLoss = []

    # loop over our data in batches
    for (batchX, batchY) in next_batch(train, probs_train, batchSize):
        if batchX.shape[0] != batchSize:
            print('Batch size insufficient (%s), continuing...' % batchY.shape[0])
            continue

        # Find current output and calculate loss for our graph
        preds = model.evaluate_total(batchX, debug=False)
        loss, error = dot_loss(preds, batchY)
        epochLoss.append(loss)

        # Update the model
        model.update(batchX, batchY, alpha)

    avloss = np.average(epochLoss)
    diff = 0.0
    if q > 0:
        # Is the fractional difference less than the threshold
        diff = math.fabs(avloss-lossHistory[-1])/avloss
        diffs.append(diff)
        losses.append(avloss)
        update = 0
        modify = True if (diff < threshold) else False
        if modify:
            # If it is less than the threshold, is it below
            # where we last updated
            modify = (avloss < (updatedLoss-(diff*avloss)))
        if modify:
            # If it is less than the threshold, is it far enough below
            # where we last updated
            modify = (avloss < (updatedLoss-(threshold*avloss)))
        if modify:
            update = 1
            print('Model conversion not sufficient, updating...')
            print('Last updated loss: %s' % updatedLoss)
            updatedLoss = avloss
            # base_pred = model.evaluate_total(train[1], debug=False)
            # model.add_layer_dynamic()
            model.expand_layer_dynamic(0)
            # print('LAYER ADDITION CHECK:')
            # pred_new = model.evaluate_total(train[1], debug=False)
            # print(dot_loss(base_pred, pred_new))
        updates.append(update)

    print('Epoch: %s, loss %s, diff %.5f, last updated loss %.5f' % (q, avloss, diff, updatedLoss))
    # update our loss history list by taking the average loss
    # across all batches
    lossHistory.append(avloss)

f_train = open(fname('', 'training_gpd', alpha, num_epochs, threshold, 'pkl'), 'wb')
training_data = [losses, diffs, updates]
pickle.dump(training_data, f_train)
f_train.close()

# plot the loss history
fig = plt.figure()
plt.plot(np.arange(0, len(lossHistory)), lossHistory)
fig.suptitle("Training loss")
plt.xlabel("Epoch num.")
plt.xlim([2, len(lossHistory)])
plt.ylim([0.0, lossHistory[2]])
plt.ylabel("Loss")
plt.yscale('log')
plt.savefig(fname("plots_gpd/", "approx_lossHistory_deep", alpha, num_epochs, threshold, ".pdf"))

# See how the sample performs on the test
testDataSig = []
testDataBkg = []
test_probabilitiesSig = []
test_probabilitiesBkg = []
for s in sig_test:
    s = scaler.transform([s])
    # prob = classifier.predict_proba(s)[0][0]
    prob = classifier.predict_proba(np.expand_dims(s, axis = 2))[0][0]
    s = s[0].flatten().tolist()
    test_probabilitiesSig.append(prob)
    testDataSig.append(s)
for b in bkg_test:
    b = scaler.transform([b])
    # prob = classifier.predict_proba(b)[0][0]
    prob = classifier.predict_proba(np.expand_dims(np.expand_dims(s, axis = 2), axis = 0))[0][0]
    b = b[0].flatten().tolist()
    test_probabilitiesBkg.append(prob)
    testDataBkg.append(b)

test = testDataSig+testDataBkg
probs_test = test_probabilitiesSig+test_probabilitiesBkg

test = np.array(test)
probs_test = np.array(probs_test)

# Make a vector of outputs
comp_preds = []
comp_true = []
for (batchX, batchY) in next_batch(test, probs_test, batchSize):
    if batchY.shape[0] < batchSize:
        print('Batch size insufficient (%s), continuing...' % batchY.shape[0])
        continue

    output = model.evaluate_total(batchX, debug=False)

    comp_preds.extend(output.T)
    comp_true.extend(batchY)
    if len(comp_preds) > 500:
        break

for n2 in range(5):
    print('predicted: %s, truth: %s' % (comp_preds[n2], comp_true[n2]))
# plot the comparison to the truth
fig2 = plt.figure()
plt.scatter(comp_preds, comp_true)
plt.xlim([0.0, 1.00])
plt.ylim([0.0, 1.00])
plt.plot([0.0, 1.0], [0.0, 1.0], 'k-')
fig.suptitle("Approximation comparison")
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.savefig(fname("plots_gpd/", "approx_vs_truth_deep", alpha, num_epochs, threshold, ".pdf"))

model.save_model(fname('', 'approx_gpd', alpha, num_epochs, threshold, '.pkl'))
