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
`model_type` : dnn  
`units` : numbers of neuron of layers for DNN  
`dp` : keep rate  
`batch_size` : batch size / one training step  

### Process
To generate word dictionary
```
python process.py
```

### Train
for DNN (95%):
```
python main.py --mode train --model_type dnn
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
`process.py` : generate dict in data/word.json
`flags.py` : all setting  
`utils.py` : get date batch  
`main.py` : main function  
`model.py` : model structure  

