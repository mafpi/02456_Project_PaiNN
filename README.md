# 02456 Molecular Property Prediction using Graph Neural Networks
This repository includes the code developed for the project "Molecular Property Prediction using Graph Neural Networks" in 02456 Deep Learning. In particular, it provides code for loading and using the QM9 dataset and for post-processing [PaiNN](https://arxiv.org/pdf/2102.03150)'s atomwise predictions.


## Modules
1. `src.data.QM9DataModule`: A PyTorch Lightning datamodule that you can use for loading and processing the QM9 dataset.
2. `src.models.AtomwisePostProcessing`: A module for performing post-processing of PaiNN's atomwise predictions. These steps can be slightly tricky/confusing but are necessary to achieve good performance (compared to normal standardization of target variables.)
3. `src.data.AtomNeighbours`: A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch. (One atom is not neighbor of itself, neighbors are only in the same molelcule).
4. 'Synopsis.md':  This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks", including background, objectives (implementing PaiNN on QM9 dataset), milestones, and key references.
5.  paiNN_simple.ipynb:This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
6. paiNN_SWA.ipynb: This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.
7. paiNN_SWAG.ipynb: this notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
8. hpc_script/: scripts to run the models in HPC

## Usage
1. `src.models.PaiNN`: A module
2. `src.models.MessagePaiNN`: A module
3. `src.models.UpdatePaiNN`: A module
4. 
