#!/usr/bin/env python
# Preprocess jet data into images


import itertools
import math
import numpy as np
import pandas
import sympy as sp
import sys
import time

# pi over 2
piovtwo = np.multiply(0.5, np.pi)
# 2 * pi
twopi = np.multiply(2.0, np.pi)

# angular separation -> R = sqrt(phi^2 + eta^2)
# for R = 0.4 aroung (phi, eta) == (0, 0)
# we cut away a square with side sqrt(0.4)
rdistance = np.sqrt(0.4)


def is_string(inpt = None):
    PY3 = sys.version_info[0] == 3
    if PY3:
        string_types = str
    else:
        string_types = basestring
    if isinstance(inpt, string_types):
        return True
    else:
        return False


def list_dframes(inpt = None):
    try:  # assume DataFrame
        assert isinstance(inpt, pandas.DataFrame)
    except AssertionError as ae:
        try:  # assume not string, but list
            assert not is_string(inpt), 'object is a string, expected DataFrame'
            assert (_item for _item in inpt), 'object is not a DataFrame and not iterable'
        except AssertionError as aee:
            raise aee  # not a list
        else:  # it's a list
            try:  # assume list of DataFrames
                for test in inpt:
                    assert isinstance(test, pandas.DataFrame), 'object is iterable, but not a list of DataFrames'
            except AssertionError as aeee:
                raise aeee  # list members are not DataFrames
            else:
                return inpt
    else:  # it's a DataFrame
        return [inpt]

def get_arr(some_var):
    '''
    Checks if object is iterable
    and returns SymPy.Array with
    appropriate shape
    '''
    try:
        (_item for _item in some_var)
        return sp.Array(some_var)
    except TypeError:
        return sp.Array([some_var])


def fix_theta_sp(theta, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for accuracy
    '''
    theta = get_arr(theta)
    return (theta.applyfunc(sp.rad) if use_deg else theta).applyfunc(sp.cos).applyfunc(sp.acos)


def fix_theta(theta, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for speed
    '''
    return np.arccos(np.cos(np.deg2rad(theta) if use_deg else theta))


def calc_eta_sp(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses sympy for accuracy
    '''
    theta = fix_theta(theta, use_deg)
    tan_theta = theta.applyfunc(lambda x: x/2).applyfunc(sp.tan)
    eta = tan_theta.applyfunc(sp.log).applyfunc(lambda x: -x).applyfunc(sp.re)
    return eta


def calc_eta(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses numpy for speed
    '''
    theta = fix_theta_np(theta)
    if theta == np.deg2rad(90.0):
        return 0.0
    return (- np.log(np.tan(np.multiply(theta, 0.5))))


def inv_eta_sp(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses sympy for accuracy
    '''
    eta = get_arr(eta)
    neta = eta.applyfunc(lambda x: -x)
    exp_neta = neta.applyfunc(sp.exp)
    theta = exp_neta.applyfunc(sp.atan).applyfunc(lambda x: 2 * x)
    return (theta.applyfunc(lambda x: sp.deg(x)) if use_deg else theta)


def inv_eta(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses numpy for speed, not accurate
    '''
    if eta == 0:
        return 90.0 if use_deg else np.multiply(0.5, np.pi)
    _theta = np.multiply(2.0, np.arctan(np.exp(np.negative(eta))))
    return np.rad2deg(_theta) if use_deg else _theta


def centre_eta_sp(eta, average_eta):
    '''
    centres pseudorapidity (eta) around 0
    in the range (-inf, inf)
    note: uses sympy for accuracy
    '''
    eta = get_arr(eta)
    average_eta = get_arr(average_eta)
    centred_eta = eta.applyfunc(lambda x: x - average_eta[0])
    return np.asarray(centred_eta.applyfunc(sp.re), dtype = 'float64')


def centre_eta(eta, average_eta):
    '''
    centres pseudorapidity (eta) around 0
    in the range (-inf, inf)
    note: uses numpy for speed, less accurate
    '''
    eta = np.asarray(eta)
    average_eta = np.asarray(average_eta)
    centred_eta = np.subtract(eta, average_eta)
    return centred_eta


def centre_phi_sp(phi, average_phi, use_deg = False):
    '''
    centres phi around pi
    '''
    phi = get_arr(phi)
    average_phi = get_arr(average_phi)
    if use_deg:
        phi = phi.applyfunc(sp.rad)
        average_phi = average_phi.applyfunc(sp.rad)
    centred_phi = phi.applyfunc(lambda x: x - average_phi[0]).applyfunc(lambda x: sp.Mod(x, 2 * sp.pi)).applyfunc(sp.re)
    return np.asarray(centred_phi.applyfunc(sp.deg) if use_deg else centred_phi, dtype = 'float64')


def centre_phi(phi, average_phi):
    '''
    centres azimutal angle (phi) around 0
    in the range [0, 2*pi)
    '''
    return np.mod(np.subtract(np.asarray(phi), average_phi), twopi)


def centre_neutral_phi(row):
    return centre_phi(row['neutral_phi'], row['nAvAzi'])


def centre_charged_phi(row):
    return centre_phi(row['charged_phi'], row['hAvAzi'])


def centre_neutral_eta(row):
    return centre_eta(row['neutral_eta'], row['nAvEta'])


def centre_charged_eta(row):
    return centred_eta(row['charged_eta'], row['hAvEta'])


def get_eta_phi_vals(data = None):
    dframes = list_dframes(data)
    eta_vals = []
    eta_vals_centred = []
    phi_vals = []
    phi_vals_centred = []
    for frame in dframes:
        eta_vals = np.concatenate((eta_vals, frame.neutral_eta.values))
        eta_vals = np.concatenate((eta_vals, frame.charged_eta.values))
        eta_vals_centred = np.concatenate((eta_vals_centred, frame.neutral_eta_centred.values))
        eta_vals_centred = np.concatenate((eta_vals_centred, frame.charged_eta_centred.values))
        phi_vals = np.concatenate((phi_vals, frame.neutral_phi.values))
        phi_vals = np.concatenate((phi_vals, frame.charged_phi.values))
        phi_vals_centred = np.concatenate((phi_vals_centred, frame.neutral_phi_centred.values))
        phi_vals_centred = np.concatenate((phi_vals_centred, frame.charged_phi_centred.values))
    return eta_vals, eta_vals_centred, phi_vals, phi_vals_centred


def calc_min_max_eta_phi(data = None):
    dframes = list_dframes(data)
    max_eta = 0
    min_eta = 0
    max_eta_centred = 0
    min_eta_centred = 0
    max_phi = 0
    min_phi = 0
    max_phi_centred = 0
    min_phi_centred = 0
    eta_vals, eta_vals_centred, phi_vals, phi_vals_centred = get_eta_phi_vals(dframes)
    for eta_list, eta_list_centred, phi_list, phi_list_centred in itertools.zip_longest(eta_vals, eta_vals_centred, phi_vals, phi_vals_centred):
        max_eta = max(np.abs(eta_list)) if max(np.abs(eta_list)) > max_eta else max_eta
        min_eta = min(np.abs(eta_list)) if min(np.abs(eta_list)) < min_eta else min_eta
        max_eta_centred = max(np.abs(eta_list_centred)) if max(np.abs(eta_list_centred)) > max_eta_centred else max_eta_centred
        min_eta_centred = min(np.abs(eta_list_centred)) if min(np.abs(eta_list_centred)) < min_eta_centred else min_eta_centred
        max_phi = max(np.abs(phi_list)) if max(np.abs(phi_list)) > max_phi else max_phi
        min_phi = min(np.abs(phi_list)) if min(np.abs(phi_list)) < min_phi else min_phi
        max_phi_centred = max(np.abs(phi_list_centred)) if max(np.abs(phi_list_centred)) > max_phi_centred else max_phi_centred
        min_phi_centred = min(np.abs(phi_list_centred)) if min(np.abs(phi_list_centred)) < min_phi_centred else min_phi_centred
    return max_eta, min_eta, max_eta_centred, min_eta_centred, max_phi, min_phi, max_phi_centred, min_phi_centred


def calc_dims_spread(data = None):
    dframes = list_dframes(data)
    max_eta, min_eta, max_eta_centred, min_eta_centred, max_phi, min_phi, max_phi_centred, min_phi_centred = calc_min_max_eta_phi(dframes)
    return max_eta_centred - max_eta, min_eta_centred - min_eta, max_phi_centred - max_phi, min_phi_centred - min_phi


def calc_dims_average(data = None):
    dframes = list_dframes(data)
    av_eta = np.asarray(list())
    av_eta_centred = np.asarray(list())
    av_phi = np.asarray(list())
    av_phi_centred = np.asarray(list())
    eta_vals, eta_vals_centred, phi_vals, phi_vals_centred = get_eta_phi_vals(dframes)
    for eta_list, eta_list_centred, phi_list, phi_list_centred in itertools.zip_longest(eta_vals, eta_vals_centred, phi_vals, phi_vals_centred):
        av_eta = np.append(av_eta, np.average(eta_list))
        av_eta_centred = np.append(av_eta_centred, np.average(eta_list_centred))
        av_phi = np.append(av_phi, np.average(phi_list))
        av_phi_centred = np.append(av_phi_centred, np.average(phi_list_centred))
    av_eta = np.average(av_eta)
    av_eta_centred = np.average(av_eta_centred)
    av_phi = np.average(av_phi)
    av_phi_centred = np.average(av_phi_centred)
    return av_eta, av_eta_centred, av_phi, av_phi_centred


saved = False
sig = None
bkg = None
if saved:
    sig = pandas.read_hdf('combined_sig_centred.h5', key='table')
    bkg = pandas.read_hdf('combined_bkg_centred.h5', key='table')
else:
    sig = pandas.read_hdf('combined_sig.h5')
    bkg = pandas.read_hdf('combined_bkg.h5')
    # apply function directly, slow
    # sig['neutral_phi_centred'] = sig.apply(centre_neutral_phi, axis=1)
    # instead combine similar operations for speed
    # cf, nf - charged phi, neutral phi
    # ce, ne - charged eta, neutral eta
    tempcf = []
    tempnf = []
    tempce = []
    tempne = []
    # two by two seems to be faster than all together
    for lstcf, fltcf, lstnf, fltnf in zip(sig.charged_phi.values,
                                          sig.hAvAzi.values.reshape(sig.hAvAzi.values.shape[0], 1),
                                          sig.neutral_phi.values,
                                          sig.nAvAzi.values.reshape(sig.nAvAzi.values.shape[0], 1)):
        tempcf.append(centre_phi(lstcf, fltcf))
        tempnf.append(centre_phi(lstnf, fltnf))
    for lstce, fltce, lstne, fltne in zip(sig.charged_eta.values,
                                          sig.hAvEta.values.reshape(sig.hAvEta.values.shape[0], 1),
                                          sig.neutral_eta.values,
                                          sig.nAvEta.values.reshape(sig.nAvEta.values.shape[0], 1)):
        tempce.append(centre_eta(lstce, fltce))
        tempne.append(centre_eta(lstne, fltne))
    # update DataFrame with new column and correlate index
    sig = sig.assign(charged_phi_centred = pandas.Series(tempcf, index = sig.index))
    sig = sig.assign(neutral_phi_centred = pandas.Series(tempnf, index = sig.index))
    sig = sig.assign(charged_eta_centred = pandas.Series(tempce, index = sig.index))
    sig = sig.assign(neutral_eta_centred = pandas.Series(tempne, index = sig.index))
    # do the same for bkg
    tempcf = []
    tempnf = []
    tempce = []
    tempne = []
    for lstcf, fltcf, lstnf, fltnf in zip(bkg.charged_phi.values,
                                          bkg.hAvAzi.values.reshape(bkg.hAvAzi.values.shape[0], 1),
                                          bkg.neutral_phi.values,
                                          bkg.nAvAzi.values.reshape(bkg.nAvAzi.values.shape[0], 1)):
        tempcf.append(centre_phi(lstcf, fltcf))
        tempnf.append(centre_phi(lstnf, fltnf))
    for lstce, fltce, lstne, fltne in zip(bkg.charged_eta.values,
                                          bkg.hAvEta.values.reshape(bkg.hAvEta.values.shape[0], 1),
                                          bkg.neutral_eta.values,
                                          bkg.nAvEta.values.reshape(bkg.nAvEta.values.shape[0], 1)):
        tempce.append(centre_eta(lstce, fltce))
        tempne.append(centre_eta(lstne, fltne))
    bkg = bkg.assign(charged_phi_centred = pandas.Series(tempcf, index = bkg.index))
    bkg = bkg.assign(neutral_phi_centred = pandas.Series(tempnf, index = bkg.index))
    bkg = bkg.assign(charged_eta_centred = pandas.Series(tempce, index = bkg.index))
    bkg = bkg.assign(neutral_eta_centred = pandas.Series(tempne, index = bkg.index))
    # save output
    sig.to_hdf('combined_sig_centred.h5', 'table')
    bkg.to_hdf('combined_bkg_centred.h5', 'table')

