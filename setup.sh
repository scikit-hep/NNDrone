#!/usr/bin/env bash

PWD=$(pwd)
export HEPDRONEHOME=${PWD} # Location of the package
export PYTHONPATH=${HEPDRONEHOME}:${PYTHONPATH}
