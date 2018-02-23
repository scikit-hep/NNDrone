#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sbenson
@author: kgizdov

"""

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pickle
from utilities.plotting import hd_hist, scatter
import matplotlib.pyplot as plt
from utilities.utilities import next_batch
from models.models import BaseModel as Model

import sys
import os, errno
from random import seed, randint
if len(sys.argv) != 2:
    print("Usage: " + str(sys.argv[0]) + " <NAME>")
    sys.exit(-1)

user_name = str(sys.argv[1])
print('User: ' + str(user_name))
print('Plots: plots/*.pdf')

def ensure_output_path(directory_path = None):
    try:
        os.makedirs(directory_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


ensure_output_path('plots')


# DATASET LOADING ***************************************************************
totalDataSig = []
probabilitiesSig = []

print 'Loading classifier...'
classifier = joblib.load("../skLearn-classifiers/classifier_rapidsim.pkl")

print 'Loading signal data file...'
sig_data = joblib.load('../data/signal_data.p')
bkg_data = joblib.load('../data/background_data.p')
#
trainFraction = 0.5
cutIndex = int(trainFraction * len(sig_data))
#
sigTrain = sig_data[: cutIndex]
sigTest = sig_data[cutIndex:]
bkgTest = bkg_data[cutIndex:]

# Create the scaler to preprocess the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(sigTrain)

sig_test = scaler.transform(sigTest)
bkg_test = scaler.transform(bkgTest)

testDataSig = []
test_probabilitiesSig = []
for s in sig_test:
    s = scaler.transform([s])
    prob = classifier.predict_proba(s)[0][0]
    s = s[0].flatten().tolist()
    test_probabilitiesSig.append(prob)
    testDataSig.append(s)

testDataBkg = []
test_probabilitiesBkg = []
for b in bkg_test:
    b = scaler.transform([b])
    prob = classifier.predict_proba(b)[0][0]
    b = b[0].flatten().tolist()
    test_probabilitiesBkg.append(prob)
    testDataBkg.append(b)

batchSize = 4
test = testDataSig
probs_test = test_probabilitiesSig
test_bkg = testDataBkg
probs_test_bkg = test_probabilitiesBkg

# Initialise model
model = Model(len(test[0]), 1)
model.load_model('../scripts/approx1.pkl')

test = np.array(test)
probs_test = np.array(probs_test)

test_bkg = np.array(test_bkg)
probs_test_bkg = np.array(probs_test_bkg)

# Make a vector of outputs
comp_preds = []
comp_true = []
for (batchX, batchY) in next_batch(test, probs_test, batchSize):
    if batchY.shape[0] < batchSize:
        print 'Batch size insufficient (%s), continuing...' % batchY.shape[0]
        continue

    output = model.evaluate_total(batchX, debug=False)

    comp_preds.extend(output.T)
    comp_true.extend(batchY)

# Make a vector of outputs
comp_preds_bkg = []
comp_true_bkg = []
for (batchX, batchY) in next_batch(test_bkg, probs_test_bkg, batchSize):
    if batchY.shape[0] < batchSize:
        print 'Batch size insufficient (%s), continuing...' % batchY.shape[0]
        continue

    output = model.evaluate_total(batchX, debug=False)

    comp_preds_bkg.extend(output.T)
    comp_true_bkg.extend(batchY)

seed(9002)
l = randint(0, len(user_name) - 2)

# plot the comparison to the truth
scatter(comp_preds, comp_true, [0.0, 1.00], [0.0, 1.00], "Prediction", "Truth " + str(user_name[l])
        , "Approximation comparison " + str(user_name[l+1]), "plots/approx_vs_truth_deep_fromLoad_" + str(user_name) + ".pdf")

comp_preds = [d.item(0) for d in comp_preds]
difflist = [(p-t) for p, t in zip(comp_preds, comp_true) if (math.fabs(p-t) < 0.0001)]
comp_preds_bkg = [d.item(0) for d in comp_preds_bkg]
difflist_bkg = [(p-t) for p, t in zip(comp_preds_bkg, comp_true_bkg) if (math.fabs(p-t) < 0.0001)]
print len(difflist)
print len(difflist_bkg)

hd_hist([difflist, difflist_bkg], 'plots/approximate_vs_truth_difference_' + str(user_name) + '.pdf'
        , [-0.00005, 0.00005], [0.0, 1100.0]
        , "Approx. difference " + str(user_name[l]), "Events " + str(user_name[l+1]), np.arange(-0.00005, 0.00005, 0.0001/350)
        , ['signal', 'background'])

# Training analysis
f_in = open('../scripts/training.pkl', 'rb')
training = pickle.load(f_in)
count = [n for n in range(len(training[0]))]
# Plot loss history
fig = plt.figure()
plt.plot(count, training[0], '-')
fig.suptitle("")
plt.xlabel("Epoch " + str(user_name[l]))
plt.ylabel("Loss " + str(user_name[l+1]))
plt.savefig("plots/loss_history_" + str(user_name) + ".pdf")
plt.clf()

# Plot difference history
fig = plt.figure()
plt.plot(count, training[1], '-')
# Add markers where an update occurs
markers_x = [c for c, a in zip(count, training[2]) if a == 1]
markers_y = [d for d, a in zip(training[1], training[2]) if a == 1]
print 'Number of model additions: %s' % len(markers_x)
plt.plot(markers_x, markers_y, 'g^')
fig.suptitle("")
plt.xlabel("Epoch " + str(user_name[l]))
plt.ylabel("Iteration difference " + str(user_name[l+1]))
plt.yscale('log')
plt.savefig("plots/difference_history_" + str(user_name) + ".pdf")
plt.clf()
