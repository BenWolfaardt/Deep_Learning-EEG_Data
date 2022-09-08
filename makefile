.ONESHELL:
SHELL=/bin/bash

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

test:
	$(CONDA_ACTIVATE) ./env
	echo $$(which python)

run_py:
	$(CONDA_ACTIVATE) ./env
	python Model_Test-Tensorflow_1.py

# Tut from: https://blog.ianpreston.ca/conda/python/bash/2020/05/13/conda_envs.html
# Interesting things: https://stackoverflow.com/questions/24736146/how-to-use-virtualenv-in-makefile

# # Choose OS setup 
# # Run scripts for requirements
# # Setup

# # Run tensorboard
# #	Either dev or local

# # .PHONY: all
# # all: ; @echo "Hos world"

# test:
# 	@echo "Hos world"

# 0:
# 	# source ~/miniforge3/bin/activate
# 	conda activate ./env

# 1:
# 	@echo "1"
# 	eval "$(conda shell.bash hook)"
# 	conda activate ./env

# 2:
# #	@echo "2"
# #	exec bash conda activate ./env
# 	source ~/miniforge3/etc/profile.d/conda.sh
# 	conda activate /Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/env
# #	conda activate ./env

# 3:
# 	@echo "3"
# #	conda init "$$SHELL"
# 	source /usr/local/opt/miniconda3/bin/activate .env/
# #	/Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/env

# 4:
# 	@echo "4"
# 	bash -c "conda activate ./env"
# # 	source ~/miniforge3/bin/activate ./env
# #	conda activate ./env

# libraries: conda
# 	@echo "Configuring environment"
# 	conda activate ./env
# #	$(shell ./setup/configure_env.sh)

# conda:
# 	@echo "Initialising and configuring Conda environment"
# 	$(shell ./setup/configure_env.sh)