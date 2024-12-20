# 02456 Molecular Property Prediction using Graph Neural Networks
## Description
This project implements the Polarizable Atom Interaction Neural Network (PaiNN) to predict molecular properties based on the QM9 dataset. The focus is on predicting the internal energy at 0K with high accuracy using rotationally equivariant graph neural networks.

## Most Notable Parts

1. **[Report](report.pdf)**: Overview of the project, this should be read first in order to have a more complete view of the code.
2. **[Simple_Painn](paiNN_simple.ipynb)**: Investigation of the paiNN model described in the original [paper](https://arxiv.org/pdf/2102.03150). Invesgtigation of effects of variation in numbers of layers
3. **[SWA](paiNN_SWA.ipynb)**: Investigation of effect of addition of stochastic weight averaging (SWA)
4. **[SWAG](paiNN_SWAG.ipynb)**: Investigation of effect of addition of stochastic weight averaging with Gaussian (SWAG)

---

## Other Modules
1. `src.data.AtomNeighbours`: A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch. (One atom is not neighbor of itself, neighbors are only in the same molelcule).
4. 'Synopsis.md':  This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks", including background, objectives (implementing PaiNN on QM9 dataset), milestones, and key references.
5.  paiNN_simple.ipynb:This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
6. paiNN_SWA.ipynb: This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.
7. paiNN_SWAG.ipynb: this notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
8. hpc_script: scripts to run the models in HPC
9. Results: results of tests
    



{In particular, it provides code for loading and using the QM9 dataset and for post-processing [PaiNN](https://arxiv.org/pdf/2102.03150)'s atomwise predictions.}

## Modules
1. [src.data.AtomNeighbours](src/data/AtomNeighbours.py): A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch. (One atom is not a neighbor of itself; neighbors are only in the same molecule).

2. [Synopsis.md](src/Synopsis.md): This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks," including background, objectives (implementing PaiNN on QM9 dataset), milestones, and key references.

3. [paiNN_simple.ipynb](src/paiNN_simple.ipynb): This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.

4. [paiNN_SWA.ipynb](src/paiNN_SWA.ipynb): This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.

5. [paiNN_SWAG.ipynb](src/paiNN_SWAG.ipynb): This notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.

6. [hpc_script](hpc_script): Scripts to run the models in HPC.

7. [Results](Results): Results of tests.

    
## Usage
