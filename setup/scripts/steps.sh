#!/bin/bash

conda create --prefix ./env python=3.8 
conda activate ./env
conda install -c apple tensorflow-deps
python -m pip install pip-tools
pip-compile --output-file=- > requirements.txt
GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 pip install grpcio==1.47.0
pip install -r requirements.txt

# ------------------------------------------------------- Notes------------------------------------------------------- #

# TODO Need to actually test this and add in -y flags

# # The below is if oyu experience some problems with grpc
# # conda install -n env/ grpc
# GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 python -m pip uninstall grpcio         
# GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 python -m pip install --no-cache-dir grpcio==1.47.0

# # v2.x + needed for Mac M1
# python -m pip uninstall tensorboard
# python -m pip install --no-cache-dir tensorboard==2.9.1
