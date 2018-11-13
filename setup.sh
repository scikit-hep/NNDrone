#!/usr/bin/env bash

if [ -z ${BASH_SOURCE+x} ]; then
    dir="$(cd -- "$(dirname -- "$0")" && pwd)"
else
    dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
fi
export NNDRONEHOME=${dir} # Location of the package
export PYTHONPATH=${NNDRONEHOME}:${PYTHONPATH}
