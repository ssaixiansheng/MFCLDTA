# MFCLDTA: Multi-scale feature contrastive learning for predicting drug-target binding affinity

MFCLDTA is an innovative model to integrate features for drugs and targets derived from three distinct scales: sequence, molecular structure,and affinity graph. An innovative multi-scale contrastive
learning framework captures the underlying relationships
among these heterogeneous information sources, yielding
enriched and robust feature representations for drug-target
pairs.

## Framework

![绘图29111](https://github.com/user-attachments/assets/23a7a2f1-2778-465c-8353-11222440ac51)

## Overview of Source Codes

- `data`: contains the input data of our model.
- `metrics.py`: contains the evaluation metrics used in our experiments.
- `GraphInput.py`: contains the construction processes of  the drug molecule graph and the target molecule graph.
- `data_preprocess.py`: contains the preprocessing of data 
- `model.py`: contains our MFCLDTA.
- `train_test.py`: contains the training and testing processes on the benchmark dataset.
- `train_test_S1.py`: contains the training and testing processes under the **S1** setting.
- `train_test_S2.py`: contains the training and testing processes under the **S2** setting.
- `train_test_S3.py`: contains the training and testing processes under the **S3** setting.
- `utils.py`: contains utility functions.

## Dependencies

- python == 3.10.11
- numpy == 1.22.4
- scikit-learn == 1.2.2
- rdkit == 2023.3.2
- networkx == 3.0
- torch == 2.0.0
- torch-geometric == 2.3.1
- lifelines == 0.27.7

## Runing

### Data Preparation

You can download the data at https://drive.google.com/file/d/1cnAeH58PhJuNiMlLqWeIxrtaqztEYHdm/view?usp=drive_link
- run data_preprocess.py: `python data_preprocess.py`

### BenchMark dataset

#### Cross Validation

Cross validation our model on the Davis dataset:

```shell
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 --fold 0
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 --fold 1
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 --fold 2
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 --fold 3
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 --fold 4

```

#### Train and Test

- Train and test our model on the Davis dataset:
    
```shell
python train_test.py --dataset davis --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 
```
    
- Train and test our model on the KIBA dataset:
    
```shell
python train_test.py --dataset kiba --cuda_id 0 --num_epochs 3000 --batch_size 512 --lr 0.0002 

```

### Cold-start scenarios
#### Setting Drug cold-start
- Train and test our model on the Davis dataset:
    
```shell
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 300 --batch_size 512 --lr 0.0005 
```
    
- Train and test our model on the KIBA dataset:
    
```shell
python train_test_S1.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 
```
    


#### Setting Target cold-start
- Train and test our model on the Davis dataset:
    
```shell
python train_test_S2.py --dataset davis --cuda_id 0 --num_epochs 300 --batch_size 512 --lr 0.0005 
 ```
    
- Train and test our model on the KIBA dataset:
    
```shell
python train_test_S2.py --dataset kiba --cuda_id 0 --num_epochs 300 --batch_size 512 --lr 0.0005 
```   
#### Setting All cold-start
- Train and test our model on the Davis dataset:
    
```shell
python train_test_S3.py --dataset davis --cuda_id 0 --num_epochs 100 --batch_size 512 --lr 0.0005 
```
    
- Train and test our model on the KIBA dataset:
    
```shell
python train_test_S3.py --dataset kiba --cuda_id 0 --num_epochs 300 --batch_size 512 --lr 0.0005 
```
The Davis dataset in the S3 experimental setting has too little data, so we use 100 epoch to prevent overfitting.
