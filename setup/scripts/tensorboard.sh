#!/bin/bash

# TODO source the activation script more generically
source /opt/miniconda3/bin/activate ./env/
wait

# Run locally
tensorboard \
  --host 0.0.0.0 \
  --port 6006 \
  --logdir logs/gradient_tape \
  serve

# Run on tensorboard.dev
# tensorboard dev upload \
#   --logdir logs/gradient_tape \
#   --name "(optional) My latest experiment" \
#   --description "(optional) Simple comparison of several hyperparameters" \
#   --one_shot
