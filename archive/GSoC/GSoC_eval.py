#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: sbenson

"""

from sklearn.externals import joblib
import numpy as np
import math

fom = lambda s, b: s/(math.sqrt(s+b))

data_path = '../GSoCeval/'
clas_name = '../GSoC/Jatin_Jindal/classifier_new.pkl'
scaler = '../skLearn-classifiers/scaler_rapidsim.pkl'
inscale = ''
if inscale:
    scaler = inscale

req_indices = [0, 1]
inputs = ['signal_data_gsoc.p', 'background_data_gsoc.p']

classifier = joblib.load(clas_name)
scaler = joblib.load(scaler)

sig_data_tot = []
print 'Loading signal data file...'
sig_data_tot = joblib.load(inputs[0])
print 'Selecting required information'
sig_data = []
for d in sig_data_tot:
    sig_data.append(np.take(d, req_indices))

bkg_data_tot = []
print 'Loading background data file...'
bkg_data_tot = joblib.load(inputs[1])
print 'Selecting required information'
bkg_data = []
for d in bkg_data_tot:
    bkg_data.append(np.take(d, req_indices))

sig_data = scaler.transform(sig_data)
bkg_data = scaler.transform(bkg_data)

passed = 0
for d in sig_data:
    scaledpoints = scaler.transform(d.reshape(1, -1))[0]
    prob = float(classifier.predict_proba([scaledpoints])[0][1])
    if prob > cuts[cl]:
        passed += 1
print 'Eff = %s' % (float(passed)/float(len(sig_data)))
