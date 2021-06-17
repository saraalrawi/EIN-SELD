#!/bin/bash

set -e
#########################################################################################################################
#CONFIG_FILE= '/home/zuern/EIN-SELD/configs/ein_seld/seld_baseline_se.yaml'
#########################################################################################################################
CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_baseline_se.yaml'

echo " Running with config: $CONFIG_FILE"
python3 -W ignore seld/main.py -c $CONFIG_FILE train --seed=$(shuf -i 0-10000 -n 1)   --num_workers=8
