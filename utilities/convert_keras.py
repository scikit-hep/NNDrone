#!/usr/bin/env python

from argparse import ArgumentParser
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import sys

from keras.layers import Conv1D, LocallyConnected1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt

try:
    from plotting import hd_hist, scatter
except ImportError:
    from utilities.plotting import hd_hist, scatter

try:
    from utilities import dot_loss, next_batch
except ImportError:
    from utilities.utilities import dot_loss, next_batch

try:
    from models import BaseModel
except ImportError:
    from models.models import BaseModel

try:
    from converters import BasicConverter
except ImportError:
    from utilities.converters import BasicConverter

# Parse arguments
parser = ArgumentParser(description = "Convert Keras model to a drone")
subparsers  = parser.add_subparsers(description = 'Choose between providing a saved Keras model or generating a new one.', dest = 'subcommand')
subparsers.required = True
parser_saved = subparsers.add_parser('saved')
parser_new   = subparsers.add_parser('new')
parser_saved.add_argument('keras_file', metavar = 'keras', action = 'store', default = 'keras_model.h5',
                          help = 'File location for saved Keras model (H5 format).')
args = parser.parse_args()

# define constants
epochNum = 1500
batchSize = 128
alpha = 0.05
threshold = 0.01

# load dataset
sig_data = np.asarray(joblib.load('../data/signal_data.p'))
bkg_data = np.asarray(joblib.load('../data/background_data.p'))

# create the scaler and transform data to be Gaussian dist with mean at 0
# important specifically for sigmoid fully connected net
scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(sig_data)
sig_data = np.asarray(scaler.transform(sig_data))
bkg_data = np.asarray(scaler.transform(bkg_data))

# define split index for dataset
trainFraction = 0.5
cutIndex = int(trainFraction * len(sig_data))

all_data = np.asarray(np.concatenate((sig_data, bkg_data)))
setTrain = np.asarray(np.concatenate((sig_data[cutIndex:],bkg_data[:cutIndex])))
setTest = np.asarray(np.concatenate((sig_data[:cutIndex],bkg_data[cutIndex:])))

if len(setTrain) != len(setTest):
    print('WARNING: Training and testing sets have different sizes: (%s; %s)' % (len(setTrain), len(setTest)))

# make labels for data
labels = np.ones(len(setTrain))
for i in range(cutIndex, len(setTrain)):
    labels[i] = 0.0  # conv nets don't like -1

useSaved = True if args.subcommand == 'saved' else False
model = None
if useSaved:
    # load Keras model
    print('Loading model')
    model = load_model(str(args.keras_file))
else:
    print('Training new model')
    ## Make Keras model
    model = Sequential()
    ## 6 inputs mean either 20 combinations of 3 classes
    model.add(LocallyConnected1D(filters = 20, kernel_size = 3, activation = 'sigmoid', input_shape = (len(sig_data[0]), 1)))
    # ## or 720 combinations of 2 classes
    # model.add(LocallyConnected1D(filters = 720, kernel_size = 2, input_shape = (len(sig_data[0]), 1)))

    # reduce spacial size of convolutional output
    # by non-linear downsampling to plug into
    # a dense layer down the chain
    model.add(GlobalMaxPooling1D())

    # match filter output number of conv layer
    model.add(Dense(30, activation = 'sigmoid'))

    # project onto 1 output
    model.add(Dense(1, activation = 'sigmoid'))

    # compile model
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    history = model.fit(np.expand_dims(setTrain, axis = 2), labels, batch_size = batchSize, epochs = epochNum, validation_data = (np.expand_dims(setTest, axis = 2), labels))

    scatter(range(0,300), history.history['loss'], [0, 300], [min(history.history['loss']), max(history.history['loss'])], 'Epoch', 'Loss', 'Training Loss', 'trainig_loss.pdf')

    joblib.dump(history.history, open('./keras_hist.pkl', 'wb'))

    model.save('./keras_locallyconnected1d_for_drone.h5')

if not model:
    print('ERROR: Could not load or create Keras model. Exiting...')
    sys.exit(1)

# get full keras response space on data
refs = []
flattened = []
for point in all_data:
    conv_point = np.expand_dims(np.expand_dims(point, axis = 2), axis = 0)
    prob = model.predict_proba(conv_point)[0][0]
    refs.append(prob)
    flattened.append(point)
refs = np.asarray(refs)
labels_ref = np.concatenate((np.ones(len(sig_data)), np.zeros(len(bkg_data))))
flattened = np.asarray(flattened)

# create drone
drone = BaseModel(len(sig_data[0]), 1)
drone.add_layer(5)
drone.add_layer(1)

conv = BasicConverter(num_epochs = epochNum, threshold = threshold)
drone = conv.convert_model(drone, model, all_data[:5000], keras_conv = True)
conv.save_history('./converted_hist.pkl')

drone.save_model('./converted_drone.pkl')

joblib.dump(scaler, open('./scaler_drone.pkl', 'wb'))

refs_drone = []
flattened_drone = []
for point in all_data:
    prob = drone.evaluate_total(point)
    refs_drone.append(prob)
    flattened_drone.append(point)
refs_drone = np.asarray(refs_drone)
labels_drone = np.concatenate((np.ones(len(sig_data)), np.zeros(len(bkg_data))))
flattened_drone = np.asarray(flattened_drone)

joblib.dump(refs, open('./response_keras.pkl', 'wb'))
joblib.dump(labels_ref, open('./labels_keras.pkl', 'wb'))
joblib.dump(refs_drone, open('./response_drone.pkl', 'wb'))
joblib.dump(labels_drone, open('./labels_drone.pkl', 'wb'))
