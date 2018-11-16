#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:44:42 2017

@author: sbenson

# Demonstrator that model conversion works
"""
import pytest
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def perform():
    from sklearn.externals import joblib
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import math
    from utilities.utilities import dot_loss, next_batch
    from nndrone.converters import BasicConverter
    from nndrone.models import BaseModel as Model
    import pickle

    # DATASET LOADING ***************************************************************
    totalDataSig = []
    totalDataBkg = []
    totalData = []
    probabilitiesSig = []
    probabilitiesBkg = []

    print ('Loading classifier...')
    classifier = joblib.load("skLearn/Type_B_MLP/Models/classifier_rapidsim.pkl")

    print ('Loading signal data file...')
    sig_data = joblib.load('data/signal_data.p')
    sig_data = sig_data[:2200]
    print ('Loading background data file...')
    bkg_data = joblib.load('data/background_data.p')
    bkg_data = bkg_data[:2200]
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
    scaler = joblib.load("skLearn/Type_B_MLP/Models/scaler_rapidsim.pkl")

    # transform the training sameple
    sig_train = scaler.transform(sigTrain)
    # do the same to the test data
    sig_test = scaler.transform(sigTest)
    # do the same to the test data
    bkg_train = scaler.transform(bgTrain)
    # do the same to the test data
    bkg_test = scaler.transform(bgTest)


    for s in sig_train:
        s = scaler.transform([s])
        prob = classifier.predict_proba(s)[0][0]
        s = s[0].flatten().tolist()
        probabilitiesSig.append(prob)
        totalDataSig.append(s)
    for b in bkg_train:
        b = scaler.transform([b])
        prob = classifier.predict_proba(b)[0][0]
        b = b[0].flatten().tolist()
        probabilitiesBkg.append(prob)
        totalDataBkg.append(b)

    # initialize a list to store the loss value for each epoch
    lossHistory = []

    # Approx. control
    alpha = 0.05
    batchSize = 10
    num_epochs = 300
    threshold = 0.02

    # layerSizes = [300, 1]
    layerSizes = [5, 1]

    converter = BasicConverter(num_epochs=num_epochs, batch_size=batchSize, learning_rate=alpha, threshold=threshold)

    train = totalDataSig+totalDataBkg
    datasize = len(train)

    # convert features and outputs to np array
    train = np.array(train)

    # shuffle signal and background
    s = np.arange(train.shape[0])
    np.random.shuffle(s)
    train = train[s]

    # Initialise model
    model = Model(len(train[0]), 1)
    if len(layerSizes) == 1:
        model.add_layer(1)
    else:
        for l in layerSizes:
            model.add_layer(l)

    model.print_layers()

    print("Starting stochastic conversion...")
    converter.convert_model(model, classifier, train)
    losses = converter.losses()
    diffs = converter.diffs()
    updates = converter.updates()

    model.save_model('approx_B_temp.pkl')

    f_train = open('approx_B_temp.pkl', 'wb')
    training_data = [losses, diffs, updates]
    pickle.dump(training_data, f_train)
    f_train.close()

    # See how the sample performs on the test
    testDataSig = []
    testDataBkg = []
    test_probabilitiesSig = []
    test_probabilitiesBkg = []
    for s in sig_test:
        s = scaler.transform([s])
        prob = classifier.predict_proba(s)[0][0]
        s = s[0].flatten().tolist()
        test_probabilitiesSig.append(prob)
        testDataSig.append(s)
    for b in bkg_test:
        b = scaler.transform([b])
        prob = classifier.predict_proba(b)[0][0]
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
            print ('Batch size insufficient (%s), continuing...' % batchY.shape[0])
            continue

        output = model.evaluate_total(batchX, debug=False)

        comp_preds.extend(output.T)
        comp_true.extend(batchY)

    acc = round(1.0-(np.mean(np.array(comp_preds)-np.array(comp_true))), 2)
    print("Accuracy = %.2f" % (1.0-(np.mean(np.array(comp_preds)-np.array(comp_true)))))
    return acc

def test_answer():
    assert perform()==0.99
