#!/bin/bash

# Install required libraries on top of existing docker image
pip install numexpr h5py numpy scipy sklearn nilearn nibabel
apt-get -y install git
pip install git+https://github.com/juhuntenburg/hcp_corr.git@enh/python3
pip install git+https://github.com/satra/mapalign.git

# run compression script
python compress_data.py

# run embedding script
python embedding_compressed.py
