# An Application of Graph Neural Network in Pollution Abatement: Acceleration Heterogeneous Catalyst Design

This repository contains the code for the paper "An Application of Graph Neural Network in Pollution Abatement: Acceleration Heterogeneous Catalyst Design".

## Introduction
This project demonstrates the application of a Graph Neural Network (GNN) in the field of pollution abatement, specifically focusing on the acceleration of heterogeneous catalyst design. The repository includes all necessary code and data to reproduce the experiments presented in the paper.

## Datasets
All datasets used for the experiments can be downloaded from the `/datasets` folder. Ensure you have the datasets in the correct path before running the preprocessing or training scripts.

## Data Preprocessing
Use dataprocess.py to preprocess the datasets. This script will prepare the data for training.

## Model Architecture
The model architecture is defined in gnn_model.py. You can review and modify this file if necessary.

## Training
To train the model, run train.py. This script will use the preprocessed data and the defined model architecture to train the GNN.

## Citation
If you find this work helpful, please consider citing it as follows:
```
@article{qiu2024application,
  title={An Application of Graph Neural Network in Pollution Abatement: Acceleration Heterogeneous Catalyst Design},
  author={Qiu, Qianhong and Du, Changming},
  journal={Materials Today Communications},
  year={2024}
}
```
