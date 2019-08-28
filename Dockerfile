# Use Python as parent image
FROM python:3.6-slim

# Install packages from pypi
RUN pip install numexpr h5py numpy scipy sklearn nilearn nibabel

# Clone and install other packages from github
RUN apt-get update && \
    apt-get -y install git && \
    pip install git+https://github.com/juhuntenburg/hcp_corr.git@enh/python3 && \
    pip install git+https://github.com/satra/mapalign.git
