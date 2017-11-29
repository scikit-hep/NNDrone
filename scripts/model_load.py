#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sbenson

"""

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from utilities.utilities import next_batch
from models.models import BaseModel as Model

# DATASET LOADING ***************************************************************
totalDataSig = []
probabilitiesSig = []

print 'Loading classifier...'
classifier = joblib.load("../skLearn-classifiers/classifier_rapidsim.pkl")

print 'Loading signal data file...'
sig_data = joblib.load('../data/signal_data.p')
#
trainFraction = 0.5
cutIndex = int(trainFraction * len(sig_data))
#
sigTrain = sig_data[: cutIndex]
sigTest = sig_data[cutIndex:]

# Create the scaler to preprocess the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(sigTrain)

sig_test = scaler.transform(sigTest)

testDataSig = []
test_probabilitiesSig = []
for s in sig_test:
    s = scaler.transform([s])
    prob = classifier.predict_proba(s)[0][0]
    s = s[0].flatten().tolist()
    test_probabilitiesSig.append(prob)
    testDataSig.append(s)

batchSize = 4
test = testDataSig
probs_test = test_probabilitiesSig

# Initialise model
model = Model(len(test[0]), 1)
model.load_model('approx1.pkl')

test = np.array(test)
probs_test = np.array(probs_test)

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

for n2 in range(5):
    print('predicted: %s, truth: %s' % (comp_preds[n2], comp_true[n2]))
# plot the comparison to the truth
fig = plt.figure()
plt.scatter(comp_preds, comp_true)
plt.xlim([0.0, 1.00])
plt.ylim([0.0, 1.00])
plt.plot([0.0, 1.0], [0.0, 1.0], 'k-')
fig.suptitle("Approximation comparison")
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.savefig("plots/approx_vs_truth_deep_fromLoad.pdf")
