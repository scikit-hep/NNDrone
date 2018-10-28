#!/usr/bin/env bash

PWD=$(pwd)
export NNDRONEHOME=${PWD} # Location of the package
export PYTHONPATH=${NNDRONEHOME}:${PYTHONPATH}
