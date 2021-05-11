#!/bin/bash

set -e
#########################################################################################################################
# Jannik, Just remove the comment ;-)
#CONFIG_FILE= '/home/zuern/EIN-SELD/configs/ein_seld/seld_baseline_mic.yaml'
#CONFIG_FILE= '/home/zuern/EIN-SELD/configs/ein_seld/seld_baseline_se_mic.yaml'
#CONFIG_FILE= '/home/zuern/EIN-SELD/configs/ein_seld/seld_attention_mic.yaml'
#########################################################################################################################
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_attention_foa.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_training_att_0.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_0.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_5.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_15.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_20.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_25.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_30.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_35.yaml'
#CONFIG_FILE='/home/alrawis/EIN-SELD/configs/ein_seld/seld_noisy_validation_att_40.yaml'

#for i in 1e-3 1e-2 1e-1
#do
  #echo " Decay value: $i"
echo " Running with config: $CONFIG_FILE"
python3 -W ignore seld/main.py -c $CONFIG_FILE train --seed=$(shuf -i 0-10000 -n 1)   --num_workers=8
#done