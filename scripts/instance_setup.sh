#!/usr/bin/env bash

apt update && apt install -y git zip && 
  git clone https://github.com/rgreenblatt/neural-render &&
  conda install -y pip scikit-image && pip install pytorch-model-summary
