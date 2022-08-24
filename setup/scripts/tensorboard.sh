#!/bin/bash

conda activate env/

# Worlks for tensorboard.dev
# tensorboard dev upload \
#   --logdir logs/gradient_tape \
#   --name "(optional) My latest experiment" \
#   --description "(optional) Simple comparison of several hyperparameters" \
#   --one_shot

# Works for local
tensorboard \
  --host 0.0.0.0 \
  --port 6006 \
  --logdir logs/gradient_tape \
  serve
