#!/usr/bin/env bash

apt update && apt install -y git && 
  git clone https://github.com/rgreenblatt/neural-render &&
  conda install -y pip && pip install pytorch-model-summary
