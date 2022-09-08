#!/bin/bash

conda create --prefix ./env python=3.9 -y
echo ""
echo "Env created, now time to install all of the libraries and dependancies"
echo ""
source /Users/james.wolfaardt/miniforge3/bin/activate ./env
conda install -c apple tensorflow-deps -y
python -m pip install pip-tools
pip-compile --output-file=- > requirements.txt
GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 pip install grpcio==1.47.0
# TODO Add step to remove grpcio from the requirements.txt so that the next step doesn't error on the  reinstall attempt
pip install -r requirements.txt
pip list