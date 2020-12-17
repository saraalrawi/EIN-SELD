#!/bin/bash

set -e

CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld.yaml'

python3 seld/main.py -c $CONFIG_FILE train --seed=$(shuf -i 0-10000 -n 1) --num_workers=8