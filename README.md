# Toxic Comment Classification Challenge
Kaggle Competition : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Preparation
```
mkdir data/
```
Download [all.zip](https://www.kaggle.com/c/8076/download-all) and put files in data/

## Model

### DNN
Fully Connected neural network  
with input : Bag of Words vector 

## Usage

### Hyperparameters in flags.py
`model_type` : DNN / CNN / RNN
`max_length` : max sentence length for CNN or RNN  
`batch_size` : batch size / one training step  
`dp` : keep rate  
`units` : numbers of neuron of layers for DNN, CNN or RNN cells  

### Process
To generate word dictionary
```
python process.py
```

### Train
for DNN (93%):
```
python main.py --mode train --model_type DNN
```
for CNN (95%):
```
python main.py --mode train --model_type CNN
```
for RNN (95%):
```
python main.py --mode train --model_type RNN
```

### Test
for output:
```
python main.py --mode test [--load (step)]
```
the output file would be data/prediction.csv

## Files

### Folders
`data/` : all data ( train.csv / test.csv )  
`model/` : store models 

### Files
`process.py` : generate dict in data/dict
`flags.py` : all setting  
`main.py` : main function  
`utils.py` : get date batch  
`model.py` : model structure  

