#!/bin/bash

# set two environment variables for git
# export GIT_COMMITTER_NAME=juhuntenburg
# export GIT_COMMITTER_EMAIL=ju.huntenburg@gmail.com

# Install required libraries on top of existing docker image
pip install --user numexpr h5py numpy scipy sklearn nilearn nibabel
pip install --user git+https://github.com/juhuntenburg/hcp_corr.git@enh/python3
pip install --user git+https://github.com/satra/mapalign.git

# run compression script
echo "Start compression"
python $@
echo "Done compression"

# run embedding script
# python embedding_compressed.py
# echo "Done embedding"
