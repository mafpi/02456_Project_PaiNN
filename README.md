# 02456 Molecular Property Prediction using Graph Neural Networks
## Description
This project explores the **Polarizable Atom Interaction Neural Network (PaiNN)** for predicting the **Internal Energy at 0K** of molecules using the **QM9 dataset**. PaiNN is a novel Graph Neural Network (GNN) that leverages rotationally equivariant representations to model molecular interactions with high accuracy. 

To enhance performance, **Stochastic Weight Averaging (SWA)** and its Gaussian variant, **SWAG**, are incorporated. We also showed that **deeper architectures** achieve better accuracy without oversmoothing.


## Most Notable Parts

1. **[report.pdf](report.pdf)**: Overview of the project, this should be read first in order to have a more complete view of the code.
   
2. **[paiNN_simple.ipynb](paiNN_simple.ipynb)**:  This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the internal energy at 0K.
   -Investigation of the paiNN model described in the original [paper](https://arxiv.org/pdf/2102.03150). Invesgtigation of effects of variation in numbers of layers.

4. **[paiNN_SWA.ipynb](paiNN_SWA.ipynb)**: This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.
   -Investigation of effect of addition of stochastic weight averaging (SWA).  

6. **[paiNN_SWAG.ipynb](paiNN_SWAG.ipynb)**:This notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
      -Investigation of effect of addition of stochastic weight averaging with Gaussian (SWAG).

## Most Notable Parts

1. **[report.pdf](report.pdf)**  
   An overview of the project. This should be read first for a complete understanding of the code and methodology.

2. **[paiNN_simple.ipynb](paiNN_simple.ipynb)**  
   Implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.  
   - Investigation of the PaiNN model described in the original [paper](https://arxiv.org/pdf/2102.03150).  
   - Examines the effects of varying the number of layers.

3. **[paiNN_SWA.ipynb](paiNN_SWA.ipynb)**  
   Implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. Includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization.  
   - Investigates the effect of adding stochastic weight averaging (SWA).

4. **[paiNN_SWAG.ipynb](paiNN_SWAG.ipynb)**  
   Implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. Includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.  
   - Investigates the effect of adding stochastic weight averaging with Gaussian (SWAG).

---

## Other Modules
1. [src.data.AtomNeighbours](src/data/AtomNeighbours.py): A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch.

2. [src.Synopsis.md](src/Synopsis.md): This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks".

3. tests

4. [hpc_script](hpc_script): Scripts to run the models on HPC.

5. [Results](Results): Results of tests on HPC.
