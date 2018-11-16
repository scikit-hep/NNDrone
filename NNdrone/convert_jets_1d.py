#!/usr/bin/env python

from argparse import ArgumentParser
from sklearn.externals import joblib
import pandas
import numpy as np
import sys

from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Activation
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping

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

try:
    from preprocessing import impose_symmetry
except ImportError:
    from utilities.preprocessing import impose_symmetry

def construct_dataset(sig):
    '''Construct Numpy Dataset from a DataFrame'''
    cols = sig.columns.values[:16]
    coll = []
    for col in cols:
        coll.append(np.asarray([np.asarray(i) for i in sig[col]]))
    coll = np.asarray(coll, dtype = 'float')
    colll = []
    for i in range(coll.shape[1]):
        colll.append(coll[:,i])
    colll = np.asarray(colll, dtype = 'float64')
    coll = colll
    # ll = [hEtProj0, hEtProj1, hMultiProj0, hMultiProj1, nEtProj0, nEtProj1, nMultiProj0, nMultiProj1]
    ll = [[], [], [], [], [], [], [], []]
    i = 0
    for i in range(8):
        if i == 0 or i == 1:
            for la in sig.charged_et_image:
                ll[i].append(np.sum(la, dtype = 'float64', axis = i % 2))
        if i == 2 or i == 3:
            for la in sig.charged_multi_image:
                ll[i].append(np.sum(la, dtype = 'float64', axis = i % 2))
        if i == 4 or i == 5:
            for la in sig.neutral_et_image:
                ll[i].append(np.sum(la, dtype = 'float64', axis = i % 2))
        if i == 6 or i == 7:
            for la in sig.neutral_multi_image:
                ll[i].append(np.sum(la, dtype = 'float64', axis = i % 2))
    for i in range(8):
        ll[i] = np.asarray(ll[i], dtype = 'float64')
    ll = np.asarray(ll, dtype = 'float64')
    colll = []
    for i in range(coll.shape[0]):
        colll.append(
            np.concatenate(
                (coll[i],
                 np.expand_dims(np.asarray(np.sum(ll[0][i]), dtype = 'float64'), axis = 0),
                 np.expand_dims(np.asarray(np.sum(ll[2][i]), dtype = 'float64'), axis = 0),
                 np.expand_dims(np.asarray(np.sum(ll[4][i]), dtype = 'float64'), axis = 0),
                 np.expand_dims(np.asarray(np.sum(ll[6][i]), dtype = 'float64'), axis = 0),
                 ll[0][i], ll[1][i],
                 ll[2][i], ll[3][i],
                 ll[4][i], ll[5][i],
                 ll[6][i], ll[7][i])
            )
        )
    colll = np.asarray(colll, dtype = 'float64')
    coll = colll
    return coll

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
    sig_data = construct_dataset(sig)
    bkg_data = construct_dataset(bkg)
else:
    sig_data = np.asarray(sig_data)
    bkg_data = np.asarray(bkg_data)

sig_data = np.expand_dims(sig_data, axis = 1)
bkg_data = np.expand_dims(bkg_data, axis = 1)

# define split index for dataset
trainFraction = 0.5
cutIndex_sig = int(trainFraction * len(sig_data))
cutIndex_bkg = int(trainFraction * len(bkg_data))

all_data = np.asarray(np.concatenate((sig_data, bkg_data)))
setTrain = np.asarray(np.concatenate((sig_data[cutIndex_sig:],bkg_data[:cutIndex_bkg])))
setTest = np.asarray(np.concatenate((sig_data[:cutIndex_sig],bkg_data[cutIndex_bkg:])))

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
    ## LocallyConnected1D to handle input of ordered 1D data
    model.add(LocallyConnected1D(filters = 32, kernel_size = 3, activation = 'relu',
                                 input_shape = sig_data[0].shape, data_format = 'channels_first'))
    # reduce spacial size of convolutional output
    # by non-linear downsampling
    model.add(MaxPooling1D(pool_size = 3, strides = 1))
    # add loss by dropout
    model.add(Dropout(0.25))
    # 2 conv layer iteration
    model.add(Conv1D(filters = 32, kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2, strides = 2))
    model.add(Dropout(0.25))
    # # 3 conv layer iteration
    # model.add(Conv1D(filters = 32, kernel_size = 2, activation = 'relu'))
    # model.add(MaxPooling1D(pool_size = 2, strides = 2))
    # model.add(Dropout(0.5))
    # flatten to feed into dense layer
    model.add(Flatten())
    # # begin down-transform for final response
    # model.add(Dense(50, activation = 'relu'))
    # project onto 1 output
    model.add(Dense(1, activation = 'sigmoid'))
    # summary
    ## model.summary()
    ## _________________________________________________________________
    ## Layer (type)                 Output Shape              Param #
    ## =================================================================
    ## locally_connected1d_10 (Loca (None, 138, 32)           17664
    ## _________________________________________________________________
    ## max_pooling1d_13 (MaxPooling (None, 136, 32)           0
    ## _________________________________________________________________
    ## dropout_12 (Dropout)         (None, 136, 32)           0
    ## _________________________________________________________________
    ## conv1d_6 (Conv1D)            (None, 134, 32)           3104
    ## _________________________________________________________________
    ## max_pooling1d_14 (MaxPooling (None, 132, 32)           0
    ## _________________________________________________________________
    ## dropout_13 (Dropout)         (None, 132, 32)           0
    ## _________________________________________________________________
    ## flatten_5 (Flatten)          (None, 4224)              0
    ## _________________________________________________________________
    ## dense_7 (Dense)              (None, 1)                 4225
    ## =================================================================
    ## Total params: 24,993
    ## Trainable params: 24,993
    ## Non-trainable params: 0
    ## _________________________________________________________________
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
    model.save('./keras_jet_conv1d_for_drone.h5')
if not model:
    # check if model does exist
    print('ERROR: Could not load or create Keras model. Exiting...')
    sys.exit(1)
# get full keras response space on data
refs = []
for point in all_data:
    prob = model.predict_proba(np.expand_dims(point, axis = 0))[0][0]
    refs.append(prob)
refs = np.asarray(refs)
labels_ref = np.concatenate((np.ones(len(sig_img)), np.zeros(len(bkg_img))))

# create drone
drone = BaseModel(len(sig_img[0].flatten()), 1)
drone.add_layer(50)
drone.add_layer(1)

conv = BasicConverter(num_epochs = epochNum, threshold = threshold)
drone = conv.convert_model(drone, model, all_data, conv_1d = True)
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
