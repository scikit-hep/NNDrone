#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sbenson

"""

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from keras.models import load_model
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
    from nndrone.models import BaseModel as Model
from utilities.utilities import scanPoint

# LOAD TEST DATA
print ('Loading signal data file...')
sig_data = joblib.load('../data/signal_data_gpd.p')
bkg_data = joblib.load('../data/background_data_gpd.p')
sig_test = sig_data[1100:2200]
bkg_test = bkg_data[1100:2200]

# LOAD MODELS
print ('Loading signal data file...')
# prefix = 'Type_B_Keras_Conv_Add_Layer_Dynamic/'
# prefix = 'Type_B_Keras_Conv/'
prefix = 'Type_GPD_Keras_Conv/'
droneLoc = prefix+'models/approx_gpd_alpha0.05_epochs1500_thresh0.02.pkl'
origLoc = prefix+'models/keras_locallyconnected1d.h5'
orig_model = load_model(origLoc)
print(len(sig_test))
drone_model = Model(len(sig_test[0]), 1)
drone_model.load_model(droneLoc)

print ('Loading scaler...')
# scaler = joblib.load("../skLearn/Type_B_MLP/Models/scaler_rapidsim.pkl")
scaler = joblib.load("../skLearn/Type_GPD_MLP/Models/scaler_rapidsim.pkl")
sig_test = scaler.transform(sig_test)
bkg_test = scaler.transform(bkg_test)

# CALCULATE RESPONSES
resp_true_sig = []
resp_drone_sig = []
for b in sig_test:
    resp_true_sig.append(orig_model.predict(np.expand_dims(np.asarray([b]), axis=2)))
    resp_drone_sig.append(drone_model.evaluate_total(np.asarray(b)))
resp_true_bkg = []
resp_drone_bkg = []
for b in bkg_test:
    resp_true_bkg.append(orig_model.predict(np.expand_dims(np.asarray([b]), axis=2)))
    resp_drone_bkg.append(drone_model.evaluate_total(np.asarray(b)))

resp_drone_sigl = [float(x) for x in resp_drone_sig]
resp_drone_bkgl = [float(x) for x in resp_drone_bkg]
resp_true_sigl = [float(x) for x in resp_true_sig]
resp_true_bkgl = [float(x) for x in resp_true_bkg]

# make ROC curves
xvals_orig = []
xvals_drone = []
yvals_orig = []
yvals_drone = []
scanpoints = np.linspace(0.0, 1.00, 5000)
for s in scanpoints:
    # print 'scanning point: %s, type: %s' % (s, type(s))
    es, rb, nSig, nBKG = scanPoint(float(s), resp_drone_sigl, resp_drone_bkgl)
    xvals_drone.append(rb)
    yvals_drone.append(es)
    es, rb, nSig, nBKG = scanPoint(float(s), resp_true_sigl, resp_true_bkgl)
    xvals_orig.append(rb)
    yvals_orig.append(es)

# Plot
fig = plt.figure()

plt.plot(xvals_drone, yvals_drone, '-b')
plt.plot(xvals_orig, yvals_orig, '-r')
fig.suptitle("")
plt.ylabel("Signal efficiency")
plt.xlabel("Background rejection")
plt.savefig(prefix+"perf/roc.pdf")
plt.clf()
