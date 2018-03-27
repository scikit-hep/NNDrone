from sklearn.externals import joblib
from array import array
import cPickle as pickle
from scipy.stats import ks_2samp
import numpy as np
import datetime
import math

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler

trainFraction = 0.5
classifier = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
                           beta_1=0.9, beta_2=0.999, early_stopping=False,
                           epsilon=1e-08, hidden_layer_sizes=(25, 20), learning_rate='adaptive',
                           learning_rate_init=0.001, max_iter=200, momentum=0.9,
                           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                           warm_start=False)


print 'Loading signal data file...'
sig_data = joblib.load('../data/signal_data.p')
print 'Loading background data file...'
bkg_data = joblib.load('../data/background_data.p')
#
cutIndex = int(trainFraction * len(sig_data))
#
sigTrain = sig_data[: cutIndex]
sigTest = sig_data[cutIndex:]
#
bgTrain = bkg_data[: cutIndex]
bgTest = bkg_data[cutIndex:]

# Create the scaler to preprocess the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(sigTrain)

# transform the training sameple
sigTrain = scaler.transform(sigTrain)
# do the same to the test data
sigTest = scaler.transform(sigTest)
# do the same to the test data
bgTrain = scaler.transform(bgTrain)
# do the same to the test data
bgTest = scaler.transform(bgTest)

print datetime.datetime.now(), 'Learning...'
train = np.append(sigTrain, bgTrain, axis=0)

target = [-1] * len(sigTrain) + [1] * len(bgTrain)
classifier.fit(train, target)

trainingSample = []
for entry in sigTrain:
    probability = float(classifier.predict_proba([entry])[0][0])
    trainingSample.append(probability)

testSample = []
for entry in sigTest:
    probability = float(classifier.predict_proba([entry])[0][0])
    testSample.append(probability)

print "Signal", ks_2samp(trainingSample, testSample)

trainingSample = []
for entry in bgTrain:
    probability = float(classifier.predict_proba([entry])[0][0])
    trainingSample.append(probability)

testSample = []
for entry in bgTest:
    probability = float(classifier.predict_proba([entry])[0][0])
    testSample.append(probability)

print "Background", ks_2samp(trainingSample, testSample)



joblib.dump(classifier, 'classifier_jindal.pkl')
print 'Classifier saved to file'
