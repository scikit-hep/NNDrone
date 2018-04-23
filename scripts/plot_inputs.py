try:
    from plotting import hd_hist
except ImportError:
    from utilities.plotting import hd_hist
from sklearn.externals import joblib
import numpy as np

ptbins = np.linspace(0.0, 10.0, num=50)
etabins = np.linspace(1.0, 6.0, num=50)

sig_data = joblib.load('../data/signal_data.p')
bkg_data = joblib.load('../data/background_data.p')

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

hd_hist([sig_pt, bkg_pt], 'plots/pt_comp.pdf'
        , [0.0, 10.0], [0.0, 1000.0]
        , "Mother $p_{T}$ GeV", "Events", ptbins
        , ['signal', 'background'])

hd_hist([sig_eta, bkg_eta], 'plots/eta_comp.pdf'
        , [1.0, 6.0], [0.0, 400.0]
        , "Mother $\eta$", "Events", etabins
        , ['signal', 'background'])

hd_hist([sig_minPT, bkg_minPT], 'plots/minpt_comp.pdf'
        , [0.0, 10.0], [0.0, 5000.0]
        , "min. $p_{T}$ GeV", "Events", ptbins
        , ['signal', 'background'])

hd_hist([sig_minETA, bkg_minETA], 'plots/mineta_comp.pdf'
        , [1.0, 6.0], [0.0, 400.0]
        , "min. $\eta$", "Events", etabins
        , ['signal', 'background'])

hd_hist([sig_maxPT, bkg_maxPT], 'plots/maxpt_comp.pdf'
        , [0.0, 10.0], [0.0, 2500.0]
        , "max. $p_{T}$ GeV", "Events", ptbins
        , ['signal', 'background'])

hd_hist([sig_maxETA, bkg_maxETA], 'plots/maxeta_comp.pdf'
        , [1.0, 6.0], [0.0, 400.0]
        , "max. $\eta$", "Events", etabins
        , ['signal', 'background'])
