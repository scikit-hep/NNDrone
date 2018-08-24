#!/usr/bin/env python
# Preprocess jet data into images


import pandas
try:
    from preprocessing import is_string, make_image, standardize_images, centre_phi, centre_eta
except ImportError:
    from utilities.preprocessing import is_string, make_image, standardize_images, centre_phi, centre_eta

# angular separation -> R = sqrt(phi^2 + eta^2)
# for R = 0.4 aroung (phi, eta) == (0, 0)
# we cut a square with side 0.4
rdistance = 0.4

def centre_jets(sig = None, bkg = None, output_filename = None):
    # applying function directly is slow
    # sig['neutral_phi_centred'] = sig.apply(centre_neutral_phi, axis = 1)
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
    if output_filename is not None:
        # save output
        output_filename = output_filename.split('.')[0]
        sig.to_hdf('sig_' + str(output_filename) + '.h5', 'table')
        bkg.to_hdf('bkg_' + str(output_filename) + '.h5', 'table')
    return sig, bkg


def image_jets(sig = None, bkg = None, save_postfix = None):
    # create charged jet images for background
    sig_charged_et_images = []
    sig_charged_multi_images = []
    for etas, phis, ets in zip(sig.charged_eta_centred, sig.charged_phi_centred, sig.charged_et):
        points = []
        for x, y, z in zip(etas, phis, ets):
            point = [x, y, z]
            points.append(point)
        charged_et_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], value_index = -1, multiplicity = False, reflect_phi_dim = 1)
        charged_multi_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], reflect_phi_dim = 1)
        sig_charged_et_images.append(charged_et_image)
        sig_charged_multi_images.append(charged_multi_image)
    # standardise images with (pixel_value - mu) / stdev
    sig_charged_et_images = standardize_images(sig_charged_et_images)
    sig_charged_multi_images = standardize_images(sig_charged_multi_images)
    # create neutral jet images for signal
    sig_neutral_et_images = []
    sig_neutral_multi_images = []
    for etas, phis, ets in zip(sig.neutral_eta_centred, sig.neutral_phi_centred, sig.neutral_et):
        points = []
        for x, y, z in zip(etas, phis, ets):
            point = [x, y, z]
            points.append(point)
        neutral_et_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], value_index = -1, multiplicity = False, reflect_phi_dim = 1)
        neutral_multi_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], reflect_phi_dim = 1)
        sig_neutral_et_images.append(neutral_et_image)
        sig_neutral_multi_images.append(neutral_multi_image)
    # standardise images
    sig_neutral_et_images = standardize_images(sig_neutral_et_images)
    sig_neutral_multi_images = standardize_images(sig_neutral_multi_images)
    # update signal DataFrame with new column and correlate index
    sig = sig.assign(charged_et_image = pandas.Series(sig_charged_et_images.tolist(), index = sig.index))
    sig = sig.assign(charged_multi_image = pandas.Series(sig_charged_multi_images.tolist(), index = sig.index))
    sig = sig.assign(neutral_et_image = pandas.Series(sig_neutral_et_images.tolist(), index = sig.index))
    sig = sig.assign(neutral_multi_image = pandas.Series(sig_neutral_multi_images.tolist(), index = sig.index))
    # create charged jet images for background
    bkg_charged_et_images = []
    bkg_charged_multi_images = []
    for etas, phis, ets in zip(bkg.charged_eta_centred, bkg.charged_phi_centred, bkg.charged_et):
        points = []
        for x, y, z in zip(etas, phis, ets):
            point = [x, y, z]
            points.append(point)
        charged_et_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], value_index = -1, multiplicity = False, reflect_phi_dim = 1)
        charged_multi_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], reflect_phi_dim = 1)
        bkg_charged_et_images.append(charged_et_image)
        bkg_charged_multi_images.append(charged_multi_image)
    # standardise images
    bkg_charged_et_images = standardize_images(bkg_charged_et_images)
    bkg_charged_multi_images = standardize_images(bkg_charged_multi_images)
    # create neutral jet images for background
    bkg_neutral_et_images = []
    bkg_neutral_multi_images = []
    for etas, phis, ets in zip(bkg.neutral_eta_centred, bkg.neutral_phi_centred, bkg.neutral_et):
        points = []
        for x, y, z in zip(etas, phis, ets):
            point = [x, y, z]
            points.append(point)
        neutral_et_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], value_index = -1, multiplicity = False, reflect_phi_dim = 1)
        neutral_multi_image = make_image(points, [[-rdistance, rdistance],[-rdistance, rdistance]], [15,15], reflect_phi_dim = 1)
        bkg_neutral_et_images.append(neutral_et_image)
        bkg_neutral_multi_images.append(neutral_multi_image)
    # standardise images
    bkg_neutral_et_images = standardize_images(bkg_neutral_et_images)
    bkg_neutral_multi_images = standardize_images(bkg_neutral_multi_images)
    # update bakground DataFrame with new column and correlate index
    bkg = bkg.assign(charged_et_image = pandas.Series(bkg_charged_et_images.tolist(), index = bkg.index))
    bkg = bkg.assign(charged_multi_image = pandas.Series(bkg_charged_multi_images.tolist(), index = bkg.index))
    bkg = bkg.assign(neutral_et_image = pandas.Series(bkg_neutral_et_images.tolist(), index = bkg.index))
    bkg = bkg.assign(neutral_multi_image = pandas.Series(bkg_neutral_multi_images.tolist(), index = bkg.index))
    if output_filename is not None:
        # save output
        output_filename = output_filename.split('.')[0]
        sig.to_hdf('sig_' + str(output_filename) + '.h5', 'table')
        bkg.to_hdf('bkg_' + str(output_filename) + '.h5', 'table')
    return sig, bkg


def process_jet_data(sig = None, bkg = None, is_centred = False, is_imaged = False, output_filename = None):
    assert sig is not None, 'please provide signal data as path or DataFrame'
    assert bkg is not None, 'please provide background data as path or DataFrame'
    if is_string(sig):
        try:
            sig = pandas.read_hdf(sig)
        except ValueError as e:
            sig = pandas.read_hdf(sig, key = 'table')
    if is_string(sig):
        try:
            bkg = pandas.read_hdf(bkg)
        except ValueError as e:
            bkg = pandas.read_hdf(bkg, key = 'table')
    assert isinstance(sig, pandas.DataFrame), 'signal object is not a DataFrame'
    assert isinstance(bkg, pandas.DataFrame), 'background object is not a DataFrame'
    if not is_centred and not is_imaged:
        sig, bkg = centre_jets(sig, bkg)
        sig, bkg = image_jets(sig, bkg)
    elif not is_imaged and is_centred:
        sig, bkg = image_jets(sig, bkg, save_postfix = output_filename)
    elif not is_centred and is_imaged:
        sig, bkg = centre_jets(sig, bkg, save_postfix = output_filename)
    else:
        return


# process the input data
process_jet_data('sig.h5', 'bkg.h5', output_filename = 'centred_imaged.h5')
