# 02456 Molecular Property Prediction using Graph Neural Networks
## Description
This project implements the Polarizable Atom Interaction Neural Network (PaiNN) to predict molecular properties based on the QM9 dataset. The focus is on predicting the internal energy at 0K with high accuracy using rotationally equivariant graph neural networks.

## Most Notable Parts

1. **[report.pdf](report.pdf)**: Overview of the project, this should be read first in order to have a more complete view of the code.
   
2. **[paiNN_simple.ipynb](paiNN_simple.ipynb)**: Investigation of the paiNN model described in the original [paper](https://arxiv.org/pdf/2102.03150). Invesgtigation of effects of variation in numbers of layers. This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.

3. **[paiNN_SWA.ipynb](paiNN_SWA.ipynb)**: Investigation of effect of addition of stochastic weight averaging (SWA).  This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.

4. **[paiNN_SWAG.ipynb](paiNN_SWAG.ipynb)**: Investigation of effect of addition of stochastic weight averaging with Gaussian (SWAG). This notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.

---

## Other Modules
1. [src.data.AtomNeighbours](src/data/AtomNeighbours.py): A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch.

2. [src.Synopsis.md](src/Synopsis.md): This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks," including background, objectives (implementing PaiNN on QM9 dataset), milestones, and key references.

3. tests

4. [hpc_script](hpc_script): Scripts to run the models on HPC.

5. [Results](Results): Results of tests on HPC.
