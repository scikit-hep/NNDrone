#!/usr/bin/env python3

try:
    from root_funcs import to_4vector, to_4vector_tree, open_root
except ImportError:
    from utilities.root_funcs import to_4vector, to_4vector_tree, open_root
from sklearn.externals import joblib

files = {'signal': ["../data/GPD/W_plusJets.root", "tree"]
    , 'background': ["../data/GPD/qcd_jets.root", "tree"]
         }

learningSetSignal = []
learningSetBackground = []

# Create signal data file
f_signal, t_signal = open_root(files['signal'][0], files['signal'][1])
entries = t_signal.GetEntries()
#
for j in range(entries):
    t_signal.GetEntry(j)
    data = list()

    data.append(t_signal.eta)
    data.append(t_signal.et)
    data.append(t_signal.m)
    data.append(t_signal.kt2)

    learningSetSignal.append(data)

    if j == 0:
        print ('Signal debug event:')
        print (data)

f_signal.Close()

joblib.dump(learningSetSignal, '../data/signal_data_gpd.p')
print ('Signal data successfully processed and saved')

# Create background data file
f_background, t_background = open_root(files['background'][0], files['background'][1])
entries = t_background.GetEntries()
#
for j in range(entries):
    t_background.GetEntry(j)
    data = list()

    data.append(t_background.eta)
    data.append(t_background.et)
    data.append(t_background.m)
    data.append(t_background.kt2)

    learningSetBackground.append(data)
    if j == 0:
        print ('Background debug event:')
        print (data)

f_background.Close()

joblib.dump(learningSetBackground, '../data/background_data_gpd.p')
print ('Background data successfully processed and saved')
