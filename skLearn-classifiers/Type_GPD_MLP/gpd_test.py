#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sbenson

"""

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
try:
    from plotting import hd_hist, scatter
except ImportError:
    from utilities.plotting import hd_hist, scatter
try:
    from utilities import next_batch
except ImportError:
    from utilities.utilities import next_batch
try:
    from models import BaseModel as Model
except ImportError:
    from models.models import BaseModel as Model
from utilities.utilities import scanPoint


def fname(prefix, base, a, n, t, suff):
    return '%s%s_alpha%s_epochs%s_thresh%s.%s' % (prefix, base, a, n, t, suff)

# DATASET LOADING ***************************************************************
totalDataSig = []
probabilitiesSig = []

print 'Loading original classifier...'
classifier = joblib.load("../skLearn-classifiers/classifier_rapidsim_gpd.pkl")

print 'Loading signal data file...'
sig_data = joblib.load('../data/signal_data_gpd.p')
bkg_data = joblib.load('../data/background_data_gpd.p')
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
    s = [s]
    prob = classifier.predict_proba(s)[0][0]
    s = s[0].flatten().tolist()
    test_probabilitiesSig.append(prob)
    testDataSig.append(s)

testDataBkg = []
test_probabilitiesBkg = []
for b in bkg_test:
    b = [b]
    prob = classifier.predict_proba(b)[0][0]
    b = b[0].flatten().tolist()
    test_probabilitiesBkg.append(prob)
    testDataBkg.append(b)

batchSize = 4
test = testDataSig
probs_test = test_probabilitiesSig
test_bkg = testDataBkg
probs_test_bkg = test_probabilitiesBkg

test = np.array(test)
probs_test = np.array(probs_test)

test_bkg = np.array(test_bkg)
probs_test_bkg = np.array(probs_test_bkg)

comp_true = [x for x in probs_test]
comp_true_bkg = [x for x in probs_test_bkg]

def predsFromModel(_modelName):
    # Initialise nominal model
    model = Model(len(test[0]), 1)
    model.load_model(_modelName)

    # Make a vector of outputs
    _comp_preds = []
    for (batchX, batchY) in next_batch(test, probs_test, batchSize):
        if batchY.shape[0] < batchSize:
            print 'Batch size insufficient (%s), continuing...' % batchY.shape[0]
            continue

        output = model.evaluate_total(batchX, debug=False)

        _comp_preds.extend(output.T)

    # Make a vector of outputs
    _comp_preds_bkg = []
    for (batchX, batchY) in next_batch(test_bkg, probs_test_bkg, batchSize):
        if batchY.shape[0] < batchSize:
            print 'Batch size insufficient (%s), continuing...' % batchY.shape[0]
            continue

        output = model.evaluate_total(batchX, debug=False)

        _comp_preds_bkg.extend(output.T)
    return _comp_preds, _comp_preds_bkg
names = ['approx_gpd_alpha0.05_epochs100_thresh0.02..pkl',
         'approx_gpd_alpha0.05_epochs500_thresh0.02..pkl',
         'approx1_gpd.pkl',
         'approx_gpd_alpha0.05_epochs1500_thresh0.02..pkl']
predsSig = []
predsBkg = []
for n in names:
    sigPreds, bkgPreds = predsFromModel(n)
    predsSig.append(sigPreds)
    predsBkg.append(bkgPreds)


    # plot the comparison to the truth
    totalpreds = sigPreds+bkgPreds
    totaltrue = comp_true+comp_true_bkg
    totalpreds = totalpreds[:400]
    gen = [x for x in range(len(totalpreds))]
    np.random.shuffle(gen)
    totalpredsNew = [totalpreds[x] for x in gen]
    totaltrue = totaltrue[:400]
    totaltrueNew = [totaltrue[x] for x in gen]
    scatter(totalpredsNew, totaltrueNew, [0.0, 1.00], [0.0, 1.00], "Prediction", "Truth"
            , "Approximation comparison", "plots_gpd/approx_vs_truth_deep_fromLoad_%s.pdf"
            % n.rstrip('..pkl'))

# make ROC curves
xvals_orig = []
xvals_drone = []
yvals_orig = []
yvals_drone = []
scanpoints = np.linspace(0.0, 1.0, 500)
for p in range(len(predsSig)):
    xvals_drone_inner = []
    yvals_drone_inner = []
    for s in scanpoints:
        es, rb, nSig, nBKG = scanPoint(s, predsSig[p], predsBkg[p])
        xvals_drone_inner.append(rb)
        yvals_drone_inner.append(es)
        if p == 0:
            es, rb, nSig, nBKG = scanPoint(s, comp_true, comp_true_bkg)
            xvals_orig.append(rb)
            yvals_orig.append(es)
    xvals_drone.append(xvals_drone_inner)
    yvals_drone.append(yvals_drone_inner)
# Plot
fig = plt.figure()
for p, q in zip(xvals_drone, yvals_drone):
    plt.plot(p, q, '-b')
plt.plot(xvals_orig, yvals_orig, '-r')
fig.suptitle("")
plt.ylabel("Signal efficiency")
plt.xlabel("Background rejection")
plt.savefig("plots_gpd/rocs.pdf")
plt.clf()
'''
comp_preds = [d.item(0) for d in comp_preds]
difflist = [(p-t) for p, t in zip(comp_preds, comp_true) if (math.fabs(p-t) < 0.0001)]
comp_preds_bkg = [d.item(0) for d in comp_preds_bkg]
difflist_bkg = [(p-t) for p, t in zip(comp_preds_bkg, comp_true_bkg) if (math.fabs(p-t) < 0.0001)]
print len(difflist)
print len(difflist_bkg)
hd_hist([difflist, difflist_bkg], 'plots_gpd/approx_vs_truth_diff.pdf'
        , [-0.00005, 0.00005], [0.0, 1100.0]
        , "Approx. difference", "Events", np.arange(-0.00005, 0.00005, 0.0001/350)
        , ['signal', 'background'])

# Training analysis **************************************************************
f_in = open('training_gpd.pkl', 'rb')
training = pickle.load(f_in)
count = [n for n in range(len(training[0]))]
# Plot loss history
fig = plt.figure()
plt.plot(count, training[0], '-')
fig.suptitle("")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("plots_gpd/loss_history.pdf")
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
plt.xlabel("Epoch")
plt.ylabel("Iteration difference")
plt.yscale('log')
plt.savefig("plots_gpd/diff_history.pdf")
plt.clf()
'''
