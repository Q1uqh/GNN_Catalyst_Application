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
Qiu, Q.; Du, C. An Application of Graph Neural Network in Pollution Abatement: Acceleration Heterogeneous Catalyst Design. Mater Today Commun. 2024, 109916. https://doi.org/10.1016/j.mtcomm.2024.109916
