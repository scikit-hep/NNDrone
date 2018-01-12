#!/usr/bin/env python3
#
# Converter from Scikit-Learn MLP to Keras lwtnn JSON

'''
Copyright 2017 Konstantin Gizdov kgizdov@gmail.com

Scikit-Learn MLP model -> lwtnn converter

Convert Scikit-Learn MLP model to JSON format for Keras with lwtnn.
Similar to https://github.com/lwtnn/lwtnn - converters/keras2json.py

Only considering linear dense models

help:
    $ python converters/mlp2json.py --help

____________________________________________________________________
Variable specification file (or read below)

In additon to the standard Keras architecture and weights files, you
must provide a "variable specification" json file with the following
format:

  {
    "inputs": [
      {"name": variable_name,
       "scale": scale,
       "offset": offset,
       "default": default_value},
      ...
      ],
    "class_labels": [output_class_1_name, output_class_2_name, ...],
    "miscellaneous": {"key": "value"}
  }

where `scale` and `offset` account for any scaling and shifting to the
input variables in preprocessing. The "default" value is optional.

The "miscellaneous" object is also optional and can contain (key,
value) pairs of strings to pass to the application.

Scikit-Learn Variable Scaler

If you provide a saved Scikit-Learn StandardScaler, variable scale and
offset will be automatically set to:
    scale  = 1 / standard_deviation  (1.0 / StandardScaler.scale_)
    offset = -mean                   (StandardScaler.mean_ - Gaussian)
This might have negaitve effects if varibles differ significantly from
normally distributed. If so, please provide a "variable specification"
file.
'''

import sys
import json
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, metrics, preprocessing
from sklearn.externals import joblib


class MLP2JSON(object):
    '''Class to convert Scikit-Learn MLP model to JSON format for Keras with lwtnn'''
    def __init__(self):
        """Class definition and init"""
        # files
        self.var_spec_file_   = None  # user provided variable specification file
        self.model_pkl_file_  = None  # MLP pkl file
        self.scaler_pkl_file_ = None  # StandardScaler pkl file
        self.output_file_     = None  # Keras JSON output file

        # Keras JSON structure
        self.keras_json_ = {
                            "defaults"      : {}
                           ,"inputs"        : []
                           ,"layers"        : []
                           ,"miscellaneous" : {}
                           ,"outputs"       : []
                           }

        # options
        self.scale_vars_     = False  # whether to use StandardScaler to make new scales/offsets
        self.save_vars_json_ = False  # whether to make new variable specification file
        self.file_with_vars_ = None   # text file with input variable names
        self.var_json_file_  = None   # file to save new var JSON to

        # model
        self.arch_         = "dense"   # Scikit-Learn MLP is fully connected always
        self.activation_   = "linear"  # assume linear activation if none resolved
        self.input_names_  = None      # input names, must be ordered
        self.class_labels_ = None      # output names, must be ordered
        self.misc_         = None      # miscellaneous, as per lwtnn requirement

        # activation conversion dictionary
        self.activation_dict_ = {
                                 'identity' : 'linear'
                                ,'logistic' : 'sigmoid'
                                ,'tanh'     : 'tanh'
                                ,'relu'     : 'rectified'
                                ,'softmax'  : 'softmax'
                                }



    def run(self):
        '''Load, convert and save MLP to a JSON file'''
        print("Scikit-Learn MLP to lwtnn JSON converter")

        self._read_model_pkl()
        self._load_vars()
        self._convert_model()
        if self.save_vars_json_:
            self._save_vars_json()  # save vars, scale and offset in a JSON file
        self._dump_json()

        return


    def _read_model_pkl(self):
        '''Load MLP from *.pkl file'''
        print(" - Reading MLP from pkl file")

        self.mlp = joblib.load(self.model_pkl_file_)  # load MLP

        self.hidden_layer_sizes_ = self.mlp.hidden_layer_sizes    # tuple of hidden layer sizes, length = n_layers - 2
        self.n_hidden_layers_    = len(self.hidden_layer_sizes_)  # number of hidden layers
        self.activation_         = self.mlp.activation            # main activation function
        self.weights_            = self.mlp.coefs_                # weight vectors, length = n_layers - 1
        self.biases_             = self.mlp.intercepts_           # bias vectors, length = n_layers - 1
        self.n_layers_           = self.mlp.n_layers_             # number of layers = input + hidden + output
        self.n_outputs_          = self.mlp.n_outputs_            # number of outputs
        self.out_activation_     = self.mlp.out_activation_       # output layer activation function

        return


    def _load_vars(self):
        '''Make sure variables, weights & biases are read & setup correctly'''

        if self.var_spec_file_:
            self._read_var_spec_file()

        elif self.file_with_vars_:
            self._get_bare_var_names()
        else:
            print("ERROR: No variable info provided. Exiting...")
            sys.exit(1)

        if self.scale_vars_ or not self.var_spec_file_:  # scale when needed
            self._scale_vars()

        self.keras_json_["inputs"]        = self.var_spec_["inputs"]
        self.keras_json_["miscellaneous"] = self.misc_

        output_num = len(self.class_labels_)
        if self.n_outputs_ != output_num:
            print("ERROR:  Number of MLP outputs ({0}) does not match\n"
                  "        number of output names ({1})!\n"
                  "        Exiting...".format(self.n_outputs_, output_num))
            sys.exit(1)
        self.keras_json_["outputs"] = self.class_labels_

        return


    def _read_var_spec_file(self):
        '''Read and load the variable spec from a file'''
        print(" - Loading variable spec JSON")

        try:
            var_spec_file = open(self.var_spec_file_, 'r')  # variable specification file as per lwtnn
        except Exception as e:
            raise e
        else:
            self.var_spec_ = json.load(var_spec_file)  # read JSON to dict

            self.input_names_  = [input_["name"] for input_ in self.var_spec_["inputs"]]
            self.class_labels_ = self.var_spec_["class_labels"]
            self.misc_         = self.var_spec_["miscellaneous"]

        return

    def _get_bare_var_names(self):
        '''Load only variable names from a text file'''

        try:
            var_file = open(self.file_with_vars_, 'r').readlines()
        except Exception as e:
            print("ERROR: Could not open variable-list file for reading. Exiting...")
            raise e
        else:
            self.input_names_ = [word for line in var_file for word in line.split()]

        return


    def _scale_vars(self):
        print (" - Using StandardScaler to scale and offset inputs...")

        try:
            scaler = joblib.load(self.scaler_pkl_file_)
        except Exception as e:
            print("ERROR: Could not load StandardScaler from file. Exiting...")
            raise e
        else:
            self.var_spec_ = {
                              "inputs"        : []
                             ,"class_labels"  : self.class_labels_
                             ,"miscellaneous" : self.misc_
                             }

            # scale and offset reciprocally each input var
            for i, var in enumerate(self.input_names_):
                var_json = {
                         "name"   : var
                        ,"offset" : - scaler.mean_[i]       # Gaussin mean
                        ,"scale"  : 1.0 / scaler.scale_[i]  # 1.0 / sqrt(scaler.var_[i])
                        }
                self.var_spec_["inputs"].append(var_json)

        return


    def _convert_model(self):
        '''Convert MLP to lwtnn'''
        print(" - Converting MLP to lwtnn dictionary")

        if any([self.n_layers_ != len(elem)+1 for elem in [self.weights_, self.biases_] ]):
            print("ERROR: Number of hidden layers ({0}) does not\n"
                  "       match length of weights ({1}) or biases ({2})!\n"
                  "       Exiting...".format(self.n_layers_, len(self.weights_)+1, len(self.biases_)+1))
            sys.exit(1)

        for l in range(self.n_layers_-1):
            layer = {}
            layer["architecture"] = self.arch_
            layer["activation"]   = self.activation_dict_[self.activation_] if l != self.n_layers_-2 else self.activation_dict_[self.out_activation_]
            layer["weights"]      = self.weights_[l].T.flatten().tolist()
            layer["bias"]         = self.biases_[l].flatten().tolist()

            self.keras_json_["layers"].append(layer)

        return


    def _save_vars_json(self):
        '''Save new variable spec JSON file'''
        print (" - Saving new variable spec JSON file to {0}".format(self.var_json_file_))

        try:
            var_spec_file = open(self.var_json_file_, 'w')
        except Exception as e:
            print("ERROR: Could not save variable spec JSON file. Exiting...")
            raise e
        else:
            var_spec_file.write(json.dumps(self.var_spec_, indent = 2, sort_keys = True))
            var_spec_file.close()

        return


    def _dump_json(self):
        '''Dump Keras JSON to file'''
        print(" - Saving model to {0}".format(self.output_file_))

        try:
            keras_json_file = open(self.output_file_, 'w')
        except IOError as e:
            print("ERROR: Could not open output file. Exiting...")
            raise e
        except OSError as e:
            print("ERROR: Could not open output file. Exiting...")
            raise e
        else:
            keras_json_file.write(json.dumps(self.keras_json_, indent = 2, sort_keys = True))
            keras_json_file.close()

        return



# Convert Scikit-Learn MLP to Keras JSON
if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser(description = "Scikit-Learn MLP to lwtnn JSON converter")

    subparsers  = parser.add_subparsers(description = 'Choose between providing a variable specification file or a list of names.', dest = 'subcommand')
    subparsers.required = True
    parser_list = subparsers.add_parser('list')
    parser_spec = subparsers.add_parser('spec')

    parser.add_argument('-m', '--model', action = 'store', default = 'mlp.pkl', dest = 'model',
                        help = '*.pkl file containing saved Scikit-Learn MLP.')
    parser.add_argument('-o', '--output', action = 'store', default = 'mlp-keras.json', dest = 'output',
                        help = 'Output file to save the Keras JSON formatted MLP.')
    parser.add_argument('-s', '--scaler', action = 'store', dest = 'scaler',
                        help = '*.pkl file containing saved Scikit-Learn StandardScaler to scale inputs.')
    parser.add_argument('-j', '--save-json', action = 'store', dest = 'save_json',
                        help = 'File to save the new Variable Specification if needed')
    parser_spec.add_argument('var_spec', metavar = 'spec', action = 'store', default = 'mlp_var_spec.json',
                             help = 'Variable Specification JSON file as per lwtnn docs.')

    parser_list.add_argument('var_list', metavar = 'list', action = 'store', default = 'mlp_var_names.txt',
                             help = 'Text file with input variable names.')
    parser_list.add_argument('scaler', metavar = 'scaler', action = 'store',
                             help = '*.pkl file containing saved Scikit-Learn StandardScaler to scale inputs.')
    parser_list.add_argument('class_label', nargs = '+', action = 'store',
                             help = 'Output label(s).')

    args = parser.parse_args()


    conv = MLP2JSON()
    conv.model_pkl_file_  = args.model
    conv.output_file_     = args.output
    conv.scaler_pkl_file_ = args.scaler
    conv.scale_vars_      = True if conv.scaler_pkl_file_ is not None else False
    conv.save_vars_json_  = True if args.save_json is not None or args.subcommand == 'list' else False

    if args.subcommand == 'list':
        conv.file_with_vars_  = args.var_list
        conv.class_labels_    = args.class_label
        conv.var_json_file_ = args.save_json if args.save_json is not None else str(conv.file_with_vars_).split('.')[0] + '_gen.json'
        conv.misc_          = {"scikit-learn": "0.19.1"}
    elif args.subcommand == 'spec':
        conv.var_spec_file_   = args.var_spec
        conv.var_json_file_ = args.save_json if args.save_json is not None else str(conv.var_spec_file_).split('.')[0] + '_gen.json'

    # Run it
    conv.run()
