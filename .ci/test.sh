#!/usr/bin/env bash

NAME=$0
COMMAND=$1

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for running test"
    echo "  run      Run tests"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    conda install mkl jupyter matplotlib numpy="$NUMPY"
    if [[ "$SCIPY" == "true" ]]; then
        conda install scipy
    fi
    pip install 'pytest>=4,<5' pytest-xdist nengo=="$NENGO"
    pip install -e .
elif [[ "$COMMAND" == "run" ]]; then
    python -c "import numpy; numpy.show_config()"
    pytest nengo_spa -n 2 -v --duration 20
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
