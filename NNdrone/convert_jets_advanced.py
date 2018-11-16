#!/usr/bin/env python

from argparse import ArgumentParser
from sklearn.externals import joblib
import pandas
import numpy as np
import sys

from keras.layers import Conv2D as KConv2D, MaxPooling2D as KMaxPooling2D, Dense as KDesne, Activation as KActivation
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from NNdrone.models import AdvancedModel
from NNdrone.layers.conv2d import Conv2D
from NNdrone.layers.maxpool2d import MaxPool2D
from NNdrone.layers.flatten import Flatten
from NNdrone.layers.dense import Dense
from NNdrone.activations import relu, sigmoid

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
    from NNdrone.models import BaseModel

try:
    from converters import AdvancedConverter
except ImportError:
    from NNdrone.converters import AdvancedConverter

try:
    from preprocessing import impose_symmetry
except ImportError:
    from utilities.preprocessing import impose_symmetry

# Parse arguments
parser = ArgumentParser(description = "Convert Keras model to a drone")
parser.add_argument('-s', '--signal', action = 'store', default = '../data/Jets/sig_centred_imaged.h5',
                    dest = 'signal', help = 'pickle file containing signal data.')
parser.add_argument('-b', '--background', action = 'store', default = '../data/Jets/bkg_centred_imaged.h5',
                    dest = 'background', help = 'pickle file containing background data.')
parser.add_argument('-f', '--fully-processed', action = 'store_true', dest = 'processed',
                    help = 'Use when images do not need to be processed for symmetries.')
parser.add_argument('-n', '--numpy-format', action = 'store_true', dest = 'np_format',
                    help = 'Use when data is stored in prepared NumPy pickle files.')
parser.add_argument('-d', '--dump-processed', action = 'store_true', dest = 'dump_processed',
                    help = 'Dump processed signal and background to files (derived from input filename).')
subparsers  = parser.add_subparsers(description = 'Choose between providing a saved Keras model or generating a new one.',
                                    dest = 'subcommand')
subparsers.required = True
parser_saved = subparsers.add_parser('saved')
parser_new   = subparsers.add_parser('new')
parser_saved.add_argument('keras_file', metavar = 'keras', action = 'store', default = 'keras_model.h5',
                          help = 'File location for saved Keras model (H5 format).')
args = parser.parse_args()
if args.processed and args.dump_processed:
    raise argparse.ArgumentError('Given data is already fully processed, nothing to dump.')

# define constants
epochNum = 1500
batchSize = 128
alpha = 0.05
threshold = 0.01

# load dataset, should be already scaled and normalized
sig_data = None
sig_img = None
bkg_data = None
sig_img = None
if not args.np_format:
    try:
        sig_data = pandas.read_hdf(parser.signal)
    except ValueError as e:
        sig_data = pandas.read_hdf(parser.signal, key = 'table')
    try:
        bkg_data = pandas.read_hdf(parser.background)
    except ValueError as e:
        bkg_data = pandas.read_hdf(parser.background, key = 'table')
else:
    sig_data = joblib.load(parser.signal)
    bkg_data = joblib.load(parser.background)
if not args.processed:
    if args.np_format:
        # not implemented
        raise argparse.ArgumentError('Functionality not yet implementated.')
    else:
        cet = np.asarray([np.asarray(i) for i in sig_data.charged_et_image])
        cet = impose_symmetry(cet)
        net = np.asarray([np.asarray(i) for i in sig_data.neutral_et_image])
        net = impose_symmetry(net)
        cmu = np.asarray([np.asarray(i) for i in sig_data.charged_multi_image])
        cmu = impose_symmetry(cmu)
        sig_img = np.asarray([np.asarray((x, y, z)) for x, y, z in zip(cet, net, cmu)])
        cet = np.asarray([np.asarray(i) for i in bkg_data.charged_et_image])
        cet = impose_symmetry(cet)
        net = np.asarray([np.asarray(i) for i in bkg_data.neutral_et_image])
        net = impose_symmetry(net)
        cmu = np.asarray([np.asarray(i) for i in bkg_data.charged_multi_image])
        cmu = impose_symmetry(cmu)
        bkg_img = np.asarray([np.asarray((x, y, z)) for x, y, z in zip(cet, net, cmu)])
else:
    sig_img = np.asarray(sig_data)
    bkg_img = np.asarray(bkg_data)

# define split index for dataset
trainFraction = 0.5
cutIndex_sig = int(trainFraction * len(sig_img))
cutIndex_bkg = int(trainFraction * len(bkg_img))

all_data = np.asarray(np.concatenate((sig_img, bkg_img)))
setTrain = np.asarray(np.concatenate((sig_img[cutIndex_sig:],bkg_img[:cutIndex_bkg])))
setTest = np.asarray(np.concatenate((sig_img[:cutIndex_sig],bkg_img[cutIndex_bkg:])))

if len(setTrain) != len(setTest):
    print('WARNING: Training and testing sets have different sizes: (%s; %s)' % (len(setTrain), len(setTest)))

# make labels for data
labels = np.ones(len(setTrain))
for i in range(cutIndex_sig, len(setTrain)):
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
    model.add(KConv2D(filters = 32, kernel_size = (3, 3), activation = 'relu',
                     input_shape = sig_img[0].shape, data_format = 'channels_first'))
    # reduce spacial size of convolutional output
    # by non-linear downsampling
    model.add(KMaxPooling2D(pool_size = (2,2), strides = 1))
    # add loss by dropout
    model.add(KDropout(0.25))
    # 2 conv layer iteration
    model.add(KConv2D(filters = 32, kernel_size = (2, 2), activation = 'relu'))
    model.add(KMaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(KDropout(0.5))
    # 3 conv layer iteration
    model.add(KConv2D(filters = 32, kernel_size = (2, 2), activation = 'relu'))
    model.add(KMaxPooling2D(pool_size = (2,2), strides = 2))
    model.add(KDropout(0.5))
    # flatten to feed into dense layer
    model.add(KFlatten())
    # begin down-transform for final response
    model.add(KDense(50, activation = 'relu'))
    # project onto 1 output
    model.add(KDense(1, activation = 'sigmoid'))
    # compile model
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    earlystop = EarlyStopping(patience = 3)
    # save history of training
    history = model.fit(setTrain, labels, batch_size = batchSize, epochs = epochNum,
                        validation_data = (setTest, labels), callbacks = [earlystop])
    # show plot
    scatter(range(0, 300), history.history['loss'], [0, 300], [min(history.history['loss']),
            max(history.history['loss'])], 'Epoch', 'Loss', 'Training Loss', 'trainig_loss.pdf')
    # save history
    joblib.dump(history.history, open('./keras_hist.pkl', 'wb'))
    # save model
    model.save('./keras_jet_conv2d_for_drone.h5')
if not model:
    # check if model does exist
    print('ERROR: Could not load or create Keras model. Exiting...')
    sys.exit(1)
# get full keras response space on data
refs = []
for point in all_data:
    prob = model.predict_proba(point)[0][0]
    refs.append(prob)
refs = np.asarray(refs)
labels_ref = np.concatenate((np.ones(len(sig_img)), np.zeros(len(bkg_img))))

# create advanced drone

drone = AdvancedModel()
drone.add(Conv2D(n_filters = 10, kernel_size = (3, 3), activation = relu,
                 input_shape = (3, 15, 15), data_format = 'channels_first'))
drone.add(MaxPool2D(pool_size = (2,2), strides = 1))
drone.add(Conv2D(n_filters = 10, kernel_size = (2, 2), activation = relu))
drone.add(MaxPool2D(pool_size = (2,2), strides = 1))
drone.add(Conv2D(n_filters = 10, kernel_size = (2, 2), activation = relu))
drone.add(MaxPool2D(pool_size = (2,2), strides = 1))
drone.add(Flatten())
drone.add(Dense(50, activation = relu))
drone.add(Dense(1, activation = sigmoid))

conv = AdvancedConverter(num_epochs = epochNum, threshold = threshold, batch_size = 10)
drone = conv.convert_model(drone, model, all_data)
conv.save_history('./converted_hist.pkl')

drone.save_model('./converted_drone.pkl')

refs_drone = []
for point in all_data:
    prob = drone.evaluate_total(point)
    refs_drone.append(prob)
refs_drone = np.asarray(refs_drone)
labels_drone = np.concatenate((np.ones(len(sig_img)), np.zeros(len(bkg_img))))

joblib.dump(refs, open('./response_keras.pkl', 'wb'))
joblib.dump(labels_ref, open('./labels_keras.pkl', 'wb'))
joblib.dump(refs_drone, open('./response_drone.pkl', 'wb'))
joblib.dump(labels_drone, open('./labels_drone.pkl', 'wb'))
