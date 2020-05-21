#!/usr/bin/env bash

apt update && apt install git && 
  git clone https://github.com/rgreenblatt/neural-render &&
  conda install pip && pip install pytorch-model-summary
