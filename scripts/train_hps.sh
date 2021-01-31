#!/bin/bash

set -e

CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld.yaml'
#1e-1 , 1e-3 , 1e-5
for i in 1e-1
do
  echo " Running Conv Othogonality with the following penalty : $i"
  python3 seld/main_hps.py -c $CONFIG_FILE train --seed=$(shuf -i 0-10000 -n 1) --num_workers=4 --r=$i

done


