#!/bin/bash

# Install required libraries on top of existing docker image
pip install --user nipype

# run compression script
echo "Start script"
python -u $@ 1>&1 2>&2
echo "Done script"
