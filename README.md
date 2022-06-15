# FCTT4Rec
Fusing Collaborative Transformation with Temporal aware Target Interaction Networks for Sequential Recommendation
## fctt

This repository contains the pytorch implementation of fctt.

## Setup

Install the required packages into your python environment:

```
pip install -r requirements.txt
```

## QuickStart

In order to train fctt, direct operation run run.py

If you want to run another dataset, change training_ parser. Py file parameters


Check out `fctt /options` for all possible options.


#### Dataset

We experimented withthree datasets: **MovieLens 1M**, **MovieLens 20M**, **Amazon Beauty** 

Choose one of these for `--dataset_code` option
* ml-1m
* ml-20m
* beauty

The raw data of these datasets will be automatically downloaded to `./Data` the first time they are required.

They will be preprocessed according to the related hyperparameters and will be saved also to `./Data` for later re-use.

Note that downloading/preprocessing is done only once per every setting to save time.

If you want to change the Data folder's path from `./Data` to somewhere else (e.g. shared folder), modify `LOCAL_DATA_FOLDER` variable in `fctt/config.py`.

#### Dataloader

Data loader we use the Bert method

#### Trainer

For the training method, we use Bert's training method


Other super parameters can be changed through 'fctt/options'

#### Model
* `--max_len`: The length of any transformer-based models
* `--hidden_units`: The size of hidden dimension
* `--num_blocks`: Number of transformer layers
* `--num_heads`: Number of attention heads

#### Dataset
* `--min_rating`: Minimum rating to regard as implicit rating. Interactions whose rating is below this value will be discarded
* `--min_uc`: Discard users whose number of ratings is below this value
* `--min_sc`: Discard items whose number of ratings is below this value

#### Dataloader
* `--dataloader_output_timestamp`: If true, the dataloader outputs timestamp information
* `--train_window`: How much to slide the training window to obtain subsequences from the user's entire item sequence
* `--train_batch_size`: Batch size for training

#### Trainer
* `--device`: CPU or CUDA
* `--use_parallel`: If true, the program uses all visible cuda devices with DataParallel
* `--optimizer`: Model optimizer (SGD or Adam)
* `--lr`: Learning rate
* `--saturation_wait_epochs`: The training will stop early if validation performance doesn't improve for this number of epochs.
* `--best_metric`: This metric will be used to compare and determine the best model

#### Negative Sampler
* `--train_negative_sample_size`: Negative sample size for training
* `--test_negative_sample_size`: Negative sample size for testing


After the training is over, the results will be saved in the folder specified by the options.

More specifically, they are saved in `experiment`

### Results

In the result folder, you will find:

```
.
├── config.json
├── models
│   ├── best_model.pth
│   ├── recent_checkpoint.pth
│   └── recent_checkpoint.pth.final
├── status.txt
└── tables
    ├── test_log.csv
    ├── train_log.csv
    └── val_log.csv
```

Below are the descriptions for the contents of each file.

#### Models

* best_model.pth: state_dict of the best model
* recent_checkpoint.pth: state_dict of the model at the latest epoch
* recent_checkpoint.pth.final: state_dict of the model at the end of the training

#### Tables

* train_log.csv: training loss at every epoch
* val_log.csv: evaluation metrics for validation data at every epoch
* test_log.csv: evaluation metrics for test data at best epoch

The baseline codes were translated to PyTorch from the following repositories:
* **MARANK**: [https://github.com/voladorlu/MARank](https://github.com/voladorlu/MARank)
* **SASRec**: [https://github.com/kang205/SASRec](https://github.com/kang205/SASRec)
* **TiSASRec**: [https://github.com/JiachengLi1995/TiSASRec](https://github.com/JiachengLi1995/TiSASRec)
* **BERT4Rec**: [https://github.com/FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec)
* **MEANTIME**[https://github.com/SungMinCho/MEANTIME](https://github.com/SungMinCho/MEANTIME)
