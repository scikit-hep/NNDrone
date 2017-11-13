#!/usr/bin/env python2

from utilities.root_funcs import to_4vector, to_4vector_tree, open_root
from sklearn.externals import joblib

files = {'signal': ["../data/RapidSimGen/Bs2Jpsiphi_tree.root", "DecayTree"]
    , 'background': ["../data/RapidSimGen/D02_4pi_tree.root", "DecayTree"]
         }

learningSetSignal = []
learningSetBackground = []

# Create signal data file
f_signal, t_signal = open_root(files['signal'][0], files['signal'][1])
entries = t_signal.GetEntries()
#
for j in xrange(entries):
    t_signal.GetEntry(j)
    data = list()

    # Mother pT
    data.append(t_signal.Bs_PT)
    # Mother eta
    data.append(to_4vector(t_signal.Bs_PX, t_signal.Bs_PY, t_signal.Bs_PZ, 5.36689).Eta())

    # Create daughter 4-vectors
    d1_4 = to_4vector_tree(t_signal, 'mup', 0.10566)
    d2_4 = to_4vector_tree(t_signal, 'mum', 0.10566)
    d3_4 = to_4vector_tree(t_signal, 'Kp', 0.49368)
    d4_4 = to_4vector_tree(t_signal, 'Km', 0.49368)
    #
    dpts = [d1_4.Pt(), d2_4.Pt(), d3_4.Pt(), d4_4.Pt()]
    detas = [d1_4.Eta(), d2_4.Eta(), d3_4.Eta(), d4_4.Eta()]

    # Min/max p_T and eta
    data.append(min(dpts))
    data.append(max(dpts))
    data.append(min(detas))
    data.append(max(detas))

    learningSetSignal.append(data)

    if j == 0:
        print 'Signal debug event:'
        print data

f_signal.Close()

joblib.dump(learningSetSignal, '../data/signal_data.p')
print 'Signal data successfully processed and saved'

# Create background data file
f_background, t_background = open_root(files['background'][0], files['background'][1])
entries = t_background.GetEntries()
#
for j in xrange(entries):
    t_background.GetEntry(j)
    data = list()

    # Mother pT
    data.append(t_background.D0_0_PT)
    # Mother eta
    data.append(to_4vector(t_background.D0_0_PX, t_background.D0_0_PY, t_background.D0_0_PZ, 1.86483).Eta())

    # Create daughter 4-vectors
    d1_4 = to_4vector_tree(t_background, 'pip_0', 0.13957)
    d2_4 = to_4vector_tree(t_background, 'pim_0', 0.13957)
    d3_4 = to_4vector_tree(t_background, 'pim_1', 0.13957)
    d4_4 = to_4vector_tree(t_background, 'pim_1', 0.13957)
    #
    dpts = [d1_4.Pt(), d2_4.Pt(), d3_4.Pt(), d4_4.Pt()]
    detas = [d1_4.Eta(), d2_4.Eta(), d3_4.Eta(), d4_4.Eta()]

    # Min/max p_T and eta
    data.append(min(dpts))
    data.append(max(dpts))
    data.append(min(detas))
    data.append(max(detas))

    learningSetBackground.append(data)
    if j == 0:
        print 'Background debug event:'
        print data

f_background.Close()

joblib.dump(learningSetBackground, '../data/background_data.p')
print 'Background data successfully processed and saved'
