#!/usr/bin/env python
# Preprocess jet data into images


import itertools
import math
import numpy as np
import pandas
import sympy as sp
import time

piovtwo = np.multiply(0.5, np.pi)
twopi = np.multiply(2.0, np.pi)


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


def fix_theta(theta, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for accuracy
    '''
    theta = get_arr(theta)
    return (theta.applyfunc(sp.rad) if use_deg else theta).applyfunc(sp.cos).applyfunc(sp.acos)


def fix_theta_np(theta, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for speed
    '''
    return np.arccos(np.cos(np.deg2rad(theta) if use_deg else theta))


def calc_eta(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses sympy for accuracy
    '''
    theta = fix_theta(theta, use_deg)
    tan_theta = theta.applyfunc(lambda x: x/2).applyfunc(sp.tan)
    eta = tan_theta.applyfunc(sp.log).applyfunc(lambda x: -x).applyfunc(sp.re)
    return eta


def calc_eta_np(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses numpy for speed
    '''
    theta = fix_theta_np(theta)
    if theta == np.deg2rad(90.0):
        return 0.0
    return (- np.log(np.tan(np.multiply(theta, 0.5))))


def inv_eta(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses sympy for accuracy
    '''
    eta = get_arr(eta)
    neta = eta.applyfunc(lambda x: -x)
    exp_neta = neta.applyfunc(sp.exp)
    theta = exp_neta.applyfunc(sp.atan).applyfunc(lambda x: 2 * x)
    return (theta.applyfunc(lambda x: sp.deg(x)) if use_deg else theta)


def inv_eta_np(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses numpy for speed, not accurate
    '''
    if eta == 0:
        return 90.0 if use_deg else np.multiply(0.5, np.pi)
    _theta = np.multiply(2.0, np.arctan(np.exp(np.negative(eta))))
    return np.rad2deg(_theta) if use_deg else _theta


def centre_eta(eta, average_eta):
    '''
    centres eta around 0 by:
    - inverting eta (-inf, inf) to theta (0, pi) rad
    - translates theta so average is pi/2, at eta == 0
    - maps back to eta (-inf, inf)
    note: uses sympy for accuracy
    '''
    theta_arr = inv_eta(eta)
    theta_av = inv_eta(average_eta)
    centring_vec = theta_av.applyfunc(lambda x: x - (sp.pi / 2))
    centred_theta = theta_arr.applyfunc(lambda x: x - centring_vec[0])
    centred_eta = calc_eta(centred_theta)
    return np.asarray(centred_eta.applyfunc(sp.re), dtype = 'float64')


def centre_eta_np(eta, average_eta):
    '''
    centres eta around 0 by:
    - inverting eta (-inf, inf) to theta (0, pi) rad
    - translates theta so average is pi/2, at eta == 0
    - maps back to eta (-inf, inf)
    note: uses numpy for speed, less accurate
    '''
    theta_arr = inv_eta_np(np.asarray(eta))
    theta_av = inv_eta_np(average_eta)
    centring_vec = np.fmod(np.subtract(theta_av, np.multiply(0.5, np.pi)), np.multiply(0.5, np.pi))
    centred_theta = np.arccos(np.cos(np.subtract(theta_arr, centring_vec)))
    centred_eta = calc_eta_np(centred_theta)
    return centred_eta


def centre_phi(phi, average_phi, use_deg = False):
    '''
    centres charged phi around pi
    '''
    phi = get_arr(phi)
    average_phi = get_arr(average_phi)
    if use_deg:
        phi = phi.applyfunc(sp.rad)
        average_phi = average_phi.applyfunc(sp.rad)
    centring_vec = average_phi.applyfunc(lambda x: x - sp.pi)
    centred_phi = phi.applyfunc(lambda x: x - centring_vec[0]).applyfunc(lambda x: sp.Mod(x, 2 * sp.pi)).applyfunc(sp.re)
    return np.asarray(centred_phi.applyfunc(sp.deg) if use_deg else centred_phi, dtype = 'float64')


def centre_phi_np(phi, average_phi):
    return np.mod(np.subtract(np.asarray(phi), average_phi), 2.0 * np.pi)


def centre_neutral_phi(row):
    return centre_phi(row['neutral_phi'], row['nAvAzi'])


def centre_charged_phi(row):
    return centre_phi(row['charged_phi'], row['hAvAzi'])


def centre_neutral_eta(row):
    return centre_eta(row['neutral_eta'], row['nAvEta'])


def centre_charged_eta(row):
    return centred_eta(row['charged_eta'], row['hAvEta'])

def get_image_dims(sig, bkg):
    max_eta = 0
    min_eta = 0
    max_phi = 0
    min_phi = 0
    eta_vals = np.concatenate((
                             sig.neutral_eta_centred.values
                            ,sig.charged_eta_centred.values
                            ,bkg.neutral_eta_centred.values
                            ,bkg.charged_eta_centred.values
                        ))
    phi_vals = np.concatenate((
                             sig.neutral_phi_centred.values
                            ,sig.charged_phi_centred.values
                            ,bkg.neutral_phi_centred.values
                            ,bkg.charged_phi_centred.values
                        ))
    for eta_list, phi_list in itertools.zip_longest(eta_vals, phi_vals):
        max_eta = max(eta_list) if max(eta_list) > max_eta else max_eta
        min_eta = min(eta_list) if min(eta_list) < min_eta else min_eta
        max_phi = max(phi_list) if max(phi_list) > max_phi else max_phi
        min_phi = min(phi_list) if min(phi_list) < min_phi else min_phi
    return max_eta, min_eta, max_phi, min_phi


def get_image_dims_abs(sig, bkg):
    max_eta = 0
    min_eta = 0
    max_phi = 0
    min_phi = 0
    eta_vals = np.concatenate((
                             sig.neutral_eta.values
                            ,sig.charged_eta.values
                            ,bkg.neutral_eta.values
                            ,bkg.charged_eta.values
                        ))
    phi_vals = np.concatenate((
                             sig.neutral_phi.values
                            ,sig.charged_phi.values
                            ,bkg.neutral_phi.values
                            ,bkg.charged_phi.values
                        ))
    for eta_list, phi_list in itertools.zip_longest(eta_vals, phi_vals):
        max_eta = max(eta_list) if max(eta_list) > max_eta else max_eta
        min_eta = min(eta_list) if min(eta_list) < min_eta else min_eta
        max_phi = max(phi_list) if max(phi_list) > max_phi else max_phi
        min_phi = min(phi_list) if min(phi_list) < min_phi else min_phi
    return max_eta, min_eta, max_phi, min_phi


def get_average_dims_abs(sig, bkg):
    av_eta = np.asarray(list())
    av_phi = np.asarray(list())
    eta_vals = np.concatenate((
                             sig.neutral_eta.values
                            ,sig.charged_eta.values
                            ,bkg.neutral_eta.values
                            ,bkg.charged_eta.values
                        ))
    phi_vals = np.concatenate((
                             sig.neutral_phi.values
                            ,sig.charged_phi.values
                            ,bkg.neutral_phi.values
                            ,bkg.charged_phi.values
                        ))
    for eta_list, phi_list in itertools.zip_longest(eta_vals, phi_vals):
        av_eta = np.append(av_eta, np.average(eta_list))
        av_phi = np.append(av_phi, np.average(phi_list))
    av_eta = np.average(av_eta)
    av_phi = np.average(av_phi)
    return av_eta, av_phi


def get_average_dims(sig, bkg):
    av_eta = np.asarray(list())
    av_phi = np.asarray(list())
    eta_vals = np.concatenate((
                             sig.neutral_eta_centred.values
                            ,sig.charged_eta_centred.values
                            ,bkg.neutral_eta_centred.values
                            ,bkg.charged_eta_centred.values
                        ))
    phi_vals = np.concatenate((
                             sig.neutral_phi_centred.values
                            ,sig.charged_phi_centred.values
                            ,bkg.neutral_phi_centred.values
                            ,bkg.charged_phi_centred.values
                        ))
    for eta_list, phi_list in itertools.zip_longest(eta_vals, phi_vals):
        av_eta = np.append(av_eta, np.average(eta_list))
        av_phi = np.append(av_phi, np.average(phi_list))
    av_eta = np.average(av_eta)
    av_phi = np.average(av_phi)
    return av_eta, av_phi


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

# get_image_dims(sig, bkg)
