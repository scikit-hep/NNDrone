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
    from sklearn.neural_network import MLPClassifier
    import numpy as np
    import math
    from utilities.utilities import dot_loss, next_batch
    from nndrone.converters import BasicConverter
    from nndrone.models import BaseModel
    try:
        from plotting import hd_hist
    except ImportError:
        from utilities.plotting import hd_hist
    from sklearn.externals import joblib
    import numpy as np

    import pickle

    # DATASET LOADING ***************************************************************
    totalDataSig = []
    totalDataBkg = []
    totalData = []
    probabilitiesSig = []
    probabilitiesBkg = []

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
    sigTrain = np.array(sig_data[: cutIndex])
    sigTest = np.array(sig_data[cutIndex:])
    #
    bgTrain = np.array(bkg_data[: cutIndex])
    bgTest = np.array(bkg_data[cutIndex:])

    classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                               beta_1=0.9, beta_2=0.999, early_stopping=False,
                               epsilon=1e-08, hidden_layer_sizes=(3, 3), learning_rate='constant',
                               learning_rate_init=0.001, max_iter=200, momentum=0.9,
                               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                               warm_start=False)

    # Create the scaler to preprocess the data
    scaler = StandardScaler()
    scaler.fit(sigTrain)

    # transform the training sameple
    sig_train = scaler.transform(sigTrain)
    # do the same to the test data
    sig_test = scaler.transform(sigTest)
    # do the same to the test data
    bkg_train = scaler.transform(bgTrain)
    # do the same to the test data
    bkg_test = scaler.transform(bgTest)

    train = np.append(sig_train, bkg_train, axis=0)

    target = np.array([1] * len(sigTrain) + [0] * len(bgTrain))
    classifier.fit(train, target)
    '''
    ptbins = np.linspace(0.0, 10.0, num=50)
    etabins = np.linspace(1.0, 6.0, num=50)

    sig_pt = [e[0] for e in sig_data]
    sig_eta = [e[1] for e in sig_data]
    sig_minPT = [e[2] for e in sig_data]
    sig_maxPT = [e[3] for e in sig_data]
    sig_minETA = [e[4] for e in sig_data]
    sig_maxETA = [e[5] for e in sig_data]
    bkg_pt = [e[0] for e in bkg_data]
    bkg_eta = [e[1] for e in bkg_data]
    bkg_minPT = [e[2] for e in bkg_data]
    bkg_maxPT = [e[3] for e in bkg_data]
    bkg_minETA = [e[4] for e in bkg_data]
    bkg_maxETA = [e[5] for e in bkg_data]

    hd_hist([sig_pt, bkg_pt], 'temp_pt_comp.pdf'
            , [0.0, 10.0], [0.0, 1000.0]
            , "Mother $p_{T}$ GeV", "Events", ptbins
            , ['signal', 'background'])

    hd_hist([sig_eta, bkg_eta], 'temp_eta_comp.pdf'
            , [1.0, 6.0], [0.0, 400.0]
            , "Mother $\eta$", "Events", etabins
            , ['signal', 'background'])

    hd_hist([sig_minPT, bkg_minPT], 'temp_minpt_comp.pdf'
            , [0.0, 10.0], [0.0, 5000.0]
            , "min. $p_{T}$ GeV", "Events", ptbins
            , ['signal', 'background'])

    hd_hist([sig_minETA, bkg_minETA], 'temp_mineta_comp.pdf'
            , [1.0, 6.0], [0.0, 400.0]
            , "min. $\eta$", "Events", etabins
            , ['signal', 'background'])

    hd_hist([sig_maxPT, bkg_maxPT], 'temp_maxpt_comp.pdf'
            , [0.0, 10.0], [0.0, 2500.0]
            , "max. $p_{T}$ GeV", "Events", ptbins
            , ['signal', 'background'])

    hd_hist([sig_maxETA, bkg_maxETA], 'temp_maxeta_comp.pdf'
            , [1.0, 6.0], [0.0, 400.0]
            , "max. $\eta$", "Events", etabins
            , ['signal', 'background'])
    '''
    print ("Getting signal probs.")
    for n, s in enumerate(sig_train):
        prob = classifier.predict_proba(s.reshape(1,-1))[0][1]
        probabilitiesSig.append(prob)
        totalDataSig.append(s)
        if n%100==0:
            print (prob)
            print (classifier.predict_proba(s.reshape(1,-1)))
    print ("Getting background probs.")
    for n, b in enumerate(bkg_train):
        prob = classifier.predict_proba(b.reshape(1,-1))[0][1]
        probabilitiesBkg.append(prob)
        totalDataBkg.append(b)
        if n%100==0:
            print (prob)
            print (classifier.predict_proba(b.reshape(1,-1)))

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
    model = BaseModel(len(train[0]), 1)
    if len(layerSizes) == 1:
        model.add_layer(1)
    else:
        for l in layerSizes:
            model.add_layer(l)

    model.print_layers()

    print("Starting stochastic conversion...")
    converter.convert_model(model, train, base_model=classifier)
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
        prob = classifier.predict_proba(s.reshape(1, -1))[0][1]
        test_probabilitiesSig.append(prob)
        testDataSig.append(s)
    for b in bkg_test:
        prob = classifier.predict_proba(b.reshape(1, -1))[0][1]
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
    print ("Sample of true values:")
    print (comp_true[:5])
    print (comp_true[-5:])
    print ("Sample of drone values:")
    print (comp_preds[:5])
    print (comp_preds[-5:])
    
    print("Accuracy = %.2f" % (1.0-(np.mean(np.array(comp_preds)-np.array(comp_true)))))
    return acc

def test_answer():
    assert perform()==0.99
