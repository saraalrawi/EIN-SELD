#!/bin/bash

set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'

python3 seld/main.py -c $CONFIG_FILE infer --num_workers=4