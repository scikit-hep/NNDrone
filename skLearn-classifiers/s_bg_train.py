from ROOT import TH1F, TCanvas, gStyle, TLegend, TGraph
from sklearn.externals import joblib
from array import array
import cPickle as pickle
from scipy.stats import ks_2samp
import numpy as np
import datetime
import math

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 

'''
#layers = ( 5, 5, 5 )
#from sklearn.neural_network import MLPClassifier
#classifier = MLPClassifier( alpha = 1e-5, hidden_layer_sizes = layers )
#
#from sklearn.preprocessing import StandardScaler
#classifier = make_pipeline( StandardScaler(), classifier )

from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier()
'''
classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,5), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
#layers = ( 7, 9, 7 ) 


#classifier = MLPClassifier( alpha = 1e-5, hidden_layer_sizes = layers )

year = "2012"
print 'Loading data file...'
data = joblib.load( '%s_KstMuMu.p' % year )
data[ 0 ] = [ i for i in data[ 0 ] if not np.any( np.isnan( i ) ) ]
data[ 1 ] = data[ 1 ][ : len( data[ 0 ] ) ]

trainFraction = 0.75
#
cutIndex = int( trainFraction * len( data[ 1 ] ) )
#
sigTrain = data[ 0 ][ : cutIndex ] # Some classifiers are biased towards dominant classes, keep the sample sizes equal
sigTest  = data[ 0 ][ cutIndex : ]
#
bgTrain = data[ 1 ][ : cutIndex ]
bgTest  = data[ 1 ][ cutIndex : ]

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
train = np.append( sigTrain, bgTrain, axis = 0 )

target = [ -1 ] * len( sigTrain ) + [ 1 ] * len( bgTrain )
classifier.fit( train, target )

print datetime.datetime.now(), 'Done'#, classifier.best_params_

bins = 50
gStyle.SetOptTitle(0)
histSigTrain = TH1F( "sig_train", "Signal probability Probability Events", bins, 0, 1 + 1e-5 )
histSigTest  = TH1F( "sig_test", "Test signal Probability Events", bins, 0, 1 + 1e-5 )
histBgTrain  = TH1F( "bg_train", "Training background Probability Events", bins, 0, 1 + 1e-5 )
histBgTest   = TH1F( "bg_test", "Test background Probability Events", bins, 0, 1 + 1e-5 )

trainingSample = []
for entry in sigTrain:
    probability = float( classifier.predict_proba( [ entry ] )[ 0 ][ 0 ] )
    trainingSample.append( probability )
    histSigTrain.Fill( probability )

testSample = []
for entry in sigTest:
    probability = float( classifier.predict_proba( [ entry ] )[ 0 ][ 0 ] )
    testSample.append( probability )
    histSigTest.Fill( probability )

print "Signal", ks_2samp( trainingSample, testSample )

trainingSample = []
for entry in bgTrain:
    probability = float( classifier.predict_proba( [ entry ] )[ 0 ][ 0 ] )
    trainingSample.append( probability )
    histBgTrain.Fill( probability )

testSample = []
for entry in bgTest:
    probability = float( classifier.predict_proba( [ entry ] )[ 0 ][ 0 ] )
    testSample.append( probability )
    histBgTest.Fill( probability )

print "Background", ks_2samp( trainingSample, testSample )

canvas1 = TCanvas( 'c1', 'Signal probability', 200, 10, 700, 500 )
gStyle.SetOptStat(0)

histSigTrain.SetLineColor( 2 )
histSigTest.SetLineColor( 3 )
histSigTrain.SetMarkerColor( 2 )
histSigTest.SetMarkerColor( 3 )
histSigTrain.SetMarkerStyle( 3 )
histSigTest.SetMarkerStyle( 3 )
#
histBgTrain.SetLineColor( 6 )
histBgTest.SetLineColor( 7 )
histBgTrain.SetMarkerColor( 6 )
histBgTest.SetMarkerColor( 7 )
histBgTrain.SetMarkerStyle( 4 )
histBgTest.SetMarkerStyle( 4 )

histSigTrain.DrawNormalized( "E" )
histSigTest.DrawNormalized( "E SAME" )
histBgTrain.DrawNormalized( "E SAME" )
histBgTest.DrawNormalized( "E SAME" )

legend = TLegend(0.35,0.7,0.65,0.9)
legend.AddEntry(histSigTrain,"Training signal","l")
legend.AddEntry(histSigTest,"","l")
legend.AddEntry(histBgTrain,"","l")
legend.AddEntry(histBgTest,"","l")
legend.Draw()

canvas1.SaveAs( "event_probability_%s.pdf" % year )

trainSignalEfficiencyArr = array( 'd' )
trainBackgroundRejectionArr = array( 'd' )
testSignalEfficiencyArr = array( 'd' )
testBackgroundRejectionArr = array( 'd' )

FoMScore = array( 'd' )
FoMProb = array( 'd' )

for bgCount, sigCount, histSig, histBg, signalEfficiencyArr, backgroundRejectionArr in (
        ( len( sigTrain ), len( bgTrain ), histSigTrain, histBgTrain, trainSignalEfficiencyArr, trainBackgroundRejectionArr ),
        ( len( sigTest ), len( bgTest ), histSigTest, histBgTest, testSignalEfficiencyArr, testBackgroundRejectionArr )
        ):
    signalEfficiency = sigCount
    backgroundRejection = 0
    for i in xrange( 0, bins + 1 ): # abuse the empty underflow bin to set initial conditions
        signalEfficiency -= histSig.GetBinContent( i )
        backgroundRejection += histBg.GetBinContent( i )
        signalEfficiencyArr.append( signalEfficiency / sigCount )
        backgroundRejectionArr.append( backgroundRejection / bgCount )
        FoMProb.append( histBg.GetBinLowEdge( i + 1 ) )
        FoMScore.append( signalEfficiency / sigCount / ( math.sqrt( bgCount - backgroundRejection ) + 3./2 ) )

trainROCGraph = TGraph( bins + 1, trainSignalEfficiencyArr, trainBackgroundRejectionArr )
testROCGraph = TGraph( bins + 1, testSignalEfficiencyArr, testBackgroundRejectionArr )

FoMGraph = TGraph( bins + 1, FoMProb, FoMScore )

print "Current  train/test ROC integral:", trainROCGraph.Integral(), "/", testROCGraph.Integral()

pickle.dump( ( ( trainSignalEfficiencyArr, trainBackgroundRejectionArr ), ( testSignalEfficiencyArr, testBackgroundRejectionArr ) ), open( 'ROC.p', 'wb' ) )

canvas2 = TCanvas( 'c2', 'Classifier ROC curve', 200, 10, 700, 500 )
gStyle.SetOptStat(0)

trainROCGraph.SetLineColor( 2 )
testROCGraph.SetLineColor( 3 )

trainROCGraph.SetTitle( "Classifier ROC curve" )
trainROCGraph.Draw()
testROCGraph.Draw( "SAME" )

legend = TLegend(0.2,0.2,0.4,0.3)
legend.AddEntry(trainROCGraph,"Training data","l")
legend.AddEntry(testROCGraph,"Test data","l")
legend.Draw()

canvas2.SaveAs( "plotsTraining/ROC_%s.pdf" % year )

canvas3 = TCanvas( 'c3', 'Figure of Merit', 200, 10, 700, 500 )
FoMGraph.Draw()
canvas3.SaveAs( "plotsTraining/FoM_%s.pdf" % year )

joblib.dump( classifier, 'classifier_%s.pkl' % year )
print 'Classifier saved to file'

raw_input( 'Press return to exit...' )
