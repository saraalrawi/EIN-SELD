#!/bin/bash

set -e
CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_visual.yaml'

#for i in 1e-3 1e-2 1e-1
#do
  #echo " Decay value: $i"
echo " Running with config: $CONFIG_FILE"
python3 -W ignore seld/main.py -c $CONFIG_FILE train --seed=$(shuf -i 0-10000 -n 1)   --num_workers=8
#
#done