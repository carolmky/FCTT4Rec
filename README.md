# FCTTRec

This repository contains PyTorch implementation of FCTTRec

## Setup

Install the required packages into your python environment:

```
pip install -r requirements.txt
```

This code was tested with `python 3.6.9` on `ubuntu` with `cuda 10.1` and various types of GPUs.

## QuickStart

In order to train FCTTRec, run run.py as follows:

```
python run.py 
```

You can also apply other templates in the `templates` folder. For example,

```
python run.py --templates train_bert
```

will train **BERT4Rec** model instead of **FCTTRec**.

It is also possible to override some options with command line arguments.

Check out `fctt/options` for all possible options.

## Training

Here is a more detailed explanation of how one can train a model.

We will explain in two levels ('Big' Choices and 'Small' Choices).


### 'Big' Choices

This project is highly modularized so that any (valid) combination of `model`, `dataset`, `dataloader`, `negative_sampler` and `trainer` will run.

#### Model

Currently, this repository provides implementations of **MEANTIME** and several other baselines .

Choose one of these for `--model_code` option
* meantime
* marank
* sas
* tisas
* bert

#### Dataset

We experimented with four datasets: **MovieLens 1M**, **MovieLens 20M**, **Amazon Beauty** .

Choose one of these for `--dataset_code` option
* ml-1m
* ml-20m
* beauty

The raw data of these datasets will be automatically downloaded to `./Data` the first time they are required.

They will be preprocessed according to the related hyperparameters and will be saved also to `./Data` for later re-use.

Note that downloading/preprocessing is done only once per every setting to save time.

If you want to change the Data folder's path from `./Data` to somewhere else (e.g. shared folder), modify `LOCAL_DATA_FOLDER` variable in `meantime/config.py`.

#### Dataloader

There is a designated dataloader for each model. Choose the right one for `--dataloader_code` option:
* bert (for **BERT4Rec** , **MEANTIME** and **FCTTRec**)
* sas (for **MARank**, **SASRec** and **TiSASRec**)

The separation is due to the way each model calculates the training loss, and the information they require.

#### Trainer

There is a designated trainer for each model. Choose the right one for `--trainer_code` option
* bert (for **BERT4Rec** , **MEANTIME** and**FCTTRec**)
* sas (for **SASRec** and **TiSASRec**)
* marank (for **MARank**)

However, at this point, all trainers have the exact same implementation thanks to the abstraction given by the models.

#### Negative Sampler

There are two types of negative samplers:
* random (sample by random)
* popular (sample according to item's popularity)

Choose one for `--train_negative_sampler`(used for training) and `--test_negative_sampler`(used for evaluation).

### 'Small' Choices (Hyperparameters)

For every big choice, one can make small choices to modify the hyperparameters that are related to the big choice.

Since there are too many options, we suggest looking at `fctt/options` for complete list.

Here we will just present some of the important ones.

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


## Viewing Training Results

### Path

After the training is over, the results will be saved in the folder specified by the options.

More specifically, they are saved in `experiment_root`/`experiment_group`/`experiment_name`

For example, `train_meantime` template has

```
experiment_root: experiments
experiment_group: test
experiment_name: fctt
```

Therefore, the results will be saved in `experiments/test/fctt`.

We suggest the users to modify the `experiment_group` and `experiment_name` options to match their purpose.

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


## References

The baseline codes were translated to PyTorch from the following repositories:
* **MARANK**: [https://github.com/voladorlu/MARank](https://github.com/voladorlu/MARank)
* **SASRec**: [https://github.com/kang205/SASRec](https://github.com/kang205/SASRec)
* **TiSASRec**: [https://github.com/JiachengLi1995/TiSASRec](https://github.com/JiachengLi1995/TiSASRec)
* **BERT4Rec**: [https://github.com/FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec)
* **MEANTIME**: .[https://github.com/SungMinCho/MEANTIME](https://github.com/SungMinCho/MEANTIME)
