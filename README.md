# A-refined-maximum-predictability-for-next-location-prediction
This repository represents the implementation of the paper: A refined maximum predictability for next location prediction with fusion knowledge

## Requirements and dependencies
This code has been tested on
- Python 3.7.13, pandas 1.3.5, numpy 1.21.6

## Folder structure
The respective code files are stored in seperate modules:
- `/data_preprocessing/*`. Codes that are used for preprocessing the dataset.
- `/refined maximum predictability/*`. Codes that are used for estimating refined maximum predictability.  
- `/improved markov/*`. Codes that are used for next location prediction. 

## Reproducing models on the Foursquare NYC dataset
### 1. Preprocess the dataset
run 
```shell
    python data_preprocessing/preprocesing_p1.py
```

### 2. caculate the refined maximum predictability for improved Markov model
run 
```shell
    python refined maximum predictability/refined maximum predictability train.py
```

### 3. run the improved Markov model
run 
```shell
    python improved markov/improved_markov model_v2.py
```
