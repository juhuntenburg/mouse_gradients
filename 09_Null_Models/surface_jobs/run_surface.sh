#!/bin/bash

pip install --user numpy scipy brainsmash

echo "Start script"
python -u $@ 1>&1 2>&2
echo "Done script"
