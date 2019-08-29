#!/bin/bash

# Install required libraries on top of existing docker image
pip install --user numexpr h5py numpy scipy sklearn nilearn nibabel
pip install --user git+https://github.com/juhuntenburg/hcp_corr.git@enh/python3
pip install --user git+https://github.com/satra/mapalign.git

# run compression script
python compress_data.py

# run embedding script
python embedding_compressed.py
