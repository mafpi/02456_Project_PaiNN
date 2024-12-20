# 02456 Molecular Property Prediction using Graph Neural Networks
## Description
This project implements the Polarizable Atom Interaction Neural Network (PaiNN) to predict molecular properties based on the QM9 dataset. The focus is on predicting the internal energy at 0K with high accuracy using rotationally equivariant graph neural networks.

## Most Notable Parts

1. **[Report](report.pdf)**: overview of the project

2. **SWAG**:
   - [SWAG](paiNN_SWAG.ipynb):  Investigation of effect of stochastic weight averaging with Gaussian (SWAG)
3. **SWA**
   - [SWA](paiNN_SWA.ipynb):Investigation of effect of stochastic weight averaging (SWA)
4. **SImple_Painn**
   - [Painn_simple](paiNN_simple.ipynb):The Polarizable Atom Interaction Neural Network (PaiNN), a Graph Neural Network that leverages rotationally equivariant representations to accurately model molecular interactions. Applied to the QM9 dataset, PaiNN predicts the Internal Energy at 0K of molecules. Investigations on the effect of multiple layer


---

## Other Modules
- [Evaluation Script](scripts/evaluate_painn.py): Tests the model on held-out data.
- [Loss Visualization Tool](scripts/plot_losses.py): Plots training and validation losses.



{In particular, it provides code for loading and using the QM9 dataset and for post-processing [PaiNN](https://arxiv.org/pdf/2102.03150)'s atomwise predictions.}

## Modules
1. `src.data.QM9DataModule`: A PyTorch Lightning datamodule that you can use for loading and processing the QM9 dataset.
2. `src.models.AtomwisePostProcessing`: A module for performing post-processing of PaiNN's atomwise predictions. These steps can be slightly tricky/confusing but are necessary to achieve good performance (compared to normal standardization of target variables.)
3. `src.data.AtomNeighbours`: A module to calculate the neighborhood adjacency matrix for atoms within a certain cutoff distance in each graph of a batch. (One atom is not neighbor of itself, neighbors are only in the same molelcule).
4. 'Synopsis.md':  This file outlines the project plan for "02456 Molecular Property Prediction using Graph Neural Networks", including background, objectives (implementing PaiNN on QM9 dataset), milestones, and key references.
5.  paiNN_simple.ipynb:This notebook implements the Polarizable Atom Interaction Neural Network (PaiNN) for predicting molecular properties using the QM9 dataset. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
6. paiNN_SWA.ipynb: This notebook implements Stochastic Weight Averaging (SWA) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, with SWA applied to enhance generalization. The model predicts the "internal energy at 0K" property, leveraging rotationally equivariant graph neural networks.
7. paiNN_SWAG.ipynb: this notebook implements Stochastic Weight Averaging Gaussian (SWAG) with the PaiNN model for predicting molecular properties using the QM9 dataset. It includes preprocessing, training, and evaluation pipelines, leveraging SWAG to sample posterior weights for uncertainty estimation and improved generalization. The target property is the "internal energy at 0K," computed as a sum of atomic contributions.
8. hpc_script: scripts to run the models in HPC
9. Results: results of tests
    
## Usage
