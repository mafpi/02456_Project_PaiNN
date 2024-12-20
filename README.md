# 02456 Molecular Property Prediction using Graph Neural Networks
This repository includes the code developed for the project "Molecular Property Prediction using Graph Neural Networks" in 02456 Deep Learning. In particular, it provides code for loading and using the QM9 dataset and for post-processing [PaiNN](https://arxiv.org/pdf/2102.03150)'s atomwise predictions.


## Modules
1. `src.data.QM9DataModule`: A PyTorch Lightning datamodule that you can use for loading and processing the QM9 dataset.
2. `src.models.AtomwisePostProcessing`: A module for performing post-processing of PaiNN's atomwise predictions. These steps can be slightly tricky/confusing but are necessary to achieve good performance (compared to normal standardization of target variables.)
3. `src.data.AtomNeighbours`: A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch. (One atom is not neighbor of itself, neighbors are only in the same molelcule).


## Usage
1. `src.models.PaiNN`: A module
2. `src.models.MessagePaiNN`: A module
3. `src.models.UpdatePaiNN`: A module
4. 