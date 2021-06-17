# Sound Event Localization and Detection
This repository is built on:
[EINV2](https://github.com/yinkalario/EIN-SELD)

## Contents

- [Download Dataset](#Download-Dataset)
- [Requirements](#Requirements)
- [Preprocessing](#Preprocessing)
- [Training Procedure](#Training-Procedure)
- [Usage](#Usage)
  * [Training](#Training)
    * [Baselines](#Baselines)
    * [S&ESELD](#S&ESELD)
    * [VASELD](#VASELD)
    * [Data Augmentation](#Data-Augmentation)
    * [Robustness Against Noise](#Robustness-Against-Noise)
   * [Prediction](#Prediction)
   * [Evaluation](#Evaluation)
- [Results](#Results)
- [Reference](#Reference)

## Download Dataset

Download dataset is easy. This script will download the dataset from: for both `FOA` and `MIC` formats. The dataset will be downloded in `/dataset_root`, run the following script:

```bash
sh scripts/download_dataset.sh
```

## Requirements
We considered a convenient anaconda environment. 

1. Use `prepare_env.sh`. Note that you need to set the `anaconda_dir` in `prepare_env.sh` to your anaconda directory, then directly run

    ```bash
    sh scripts/prepare_env.sh
    ```

2. Use `environment.ymal`. Note that you also need to set the `prefix` to your aimed env directory, then directly run, `environment.yml` includes all the dependencies
consider increasing the disk space of your conda environment to avoid running out of space.  

    ```bash
    conda env create -f environment.yml
    ```
    
The conda environment named ein, activate it then you are all set: 
```bash
conda activate ein
```

## Preprocessing
The config files are saved EIN-SELD/configs/ein_seld/, you need to change to the path to your home directory.
Here the audio and metadata/labels will be processed so the  `.wav` files will be saved to `.h5` files. Meta files will also be converted to `.h5` files. Run the following script after downloading the data.

```bash
sh scripts/preproc.sh
```

Preprocessing for meta files/labels separate labels to different tracks, each with up to one event and a corresponding DoA. The same event is consistently put in the same track. The authors of EINV2 claims that this is necessary for chunk-level PIT; however, there is no code provided for chunk-level PIT and we can train our models with frame-level PIT.

## Training
The training configurations are saved in `ymal` files. Basically, we have the following networks `EINV2`, `EINV2-C`,  `SELD_ATT` and `EINV2` with `weight_sharing` configured as `attention_se` . The  `EINV2` and `EINV2-C` are the baselines, `SELD_ATT` is `VASELD` and `EINV2-C` with `weight_sharing: attention_se` is `S&ESELD`. 
To train the models, we provided indiviual config file for each model, we further use weight and biases logging tool to observe the learning curves. 
The dataset has two sets of folds: `train_fold` and `valid_fold`. We report our results on `valid_fold` and combined `overlap`, namely `1&2`. The hyper-parameter description is provided as comments in the `seld_*.yml` files. 
 
To weight and biases install run the following command in your activated `ein` conda environment. You can deactive the logging by setting `wandb_active:False` in the config file.

```bash
pip install wandb
```

### Baselines
Baseline EINV2
```bash
sh scripts/train.sh
```
Constrained Baseline EINV2-C
```bash
sh scripts/train_c.sh
```
### VASELD

```bash
sh scripts/train_visual.sh
```
### S&ESELD
```bash
sh scripts/train_se.sh
```

## Prediction
The prediction results will be provided per file for the test dataset that consists of 200 files and saved in `foa_eval` and `mic_eval` for `FOA` and `MIC` datasets, respectively. The models output will be converted to DCASE format and saved in in `/out_infer_model_name` as `csv` files.
The provided `predict.sh` is for our VASEL.

```bash
sh scripts/predict.sh
```
`visualize_prediction.py` is adapted with out parameters. To visualise the predictions, run the following: 

```python
python3 visualize_prediction.py --dataset_dir YOUR_USER_DIR/EIN-SELD/_dataset/dataset_root/  --pred_file YOUR_USER_DIR/EIN-SELD/submissions/mix001.csv --plot_loc YOUR_USER_DIR//EIN-SELD/submissions/
```
The output file will be saved as a `.jpg` in the provided location. 

## Evaluation
```bash
bash scripts/evaluate.sh
```
