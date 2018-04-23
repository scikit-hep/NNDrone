try:
    from plotting import hd_hist
except ImportError:
    from utilities.plotting import hd_hist
from sklearn.externals import joblib
import numpy as np

etbins = np.linspace(20.0, 400.0, num=100)
etabins = np.linspace(-4.0, 4.0, num=100)
mbins = np.linspace(0.0, 200.0, num=100)
ktbins = np.linspace(0.0, 100000.0, num=100)

sig_data = joblib.load('../data/signal_data_gpd.p')
bkg_data = joblib.load('../data/background_data_gpd.p')

sig_data = sig_data[:2000]
bkg_data = bkg_data[:2000]
sig_eta = [e[0] for e in sig_data]
sig_et = [e[1] for e in sig_data]
sig_m = [e[2] for e in sig_data]
sig_kt = [e[3] for e in sig_data]
bkg_eta = [e[0] for e in bkg_data]
bkg_et = [e[1] for e in bkg_data]
bkg_m = [e[2] for e in bkg_data]
bkg_kt = [e[3] for e in bkg_data]

hd_hist([sig_et, bkg_et], 'plots_gpd/et_comp_gpd.pdf'
        , [20.0, 400.0], [0.0, 600.0]
        , "$E_{T}$ GeV", "Events", etbins
        , ['signal', 'background'])

hd_hist([sig_eta, bkg_eta], 'plots_gpd/eta_comp_gpd.pdf'
        , [-4.0, 4.0], [0.0, 100.0]
        , "$\eta$", "Events", etabins
        , ['signal', 'background'])

hd_hist([sig_m, bkg_m], 'plots_gpd/m_comp_gpd.pdf'
        , [0.0, 100.0], [0.0, 600.0]
        , "Mass [GeV]", "Events", mbins
        , ['signal', 'background'])

hd_hist([sig_kt, bkg_kt], 'plots_gpd/kt_comp_gpd.pdf'
        , [0.0, 100000.0], [0.0, 1000.0]
        , "$K_{T}$", "Events", ktbins
        , ['signal', 'background'])
