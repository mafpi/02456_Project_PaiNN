# 02456 Molecular Property Prediction using Graph Neural Networks

## Background and Motivation

The accurate prediction of chemical properties and efficient simulation of molecular dynamics are critical to advancements in fields like drug discovery, materials science, and chemical engineering. Traditional computational methods, while precise, often involve complex calculations that can be prohibitively slow and computationally expensive, especially for large molecules or extensive datasets. One promising alternative is Graph Neural Networks (GNNs), which has become a dominant and fast-growing method in deep learning with graph data [[3]](#3).

Molecules have specific configurations that can be represented as graphs of atoms and bounds. These structures can be handled by GNNs that preserve molecular properties and structures. However, given that standard message passing formulations do not consider rotationally equivariant representations, Kristof T. Schütt, et al. ([[1]](#1)), proposed a different approach: Polarizable Atom Interaction Neural Networn (PaiNN). This approach captures geometric information and physical interactions, as it takes into account rotational equivariance of molecules.

Therefore, the main goal of this project is to implement PaiNN and apply it to molecular property prediction using the QM9 dataset, which includes geometric, energetic, electronic and thermodynamic properties of 134k small stable organic molecules made up of CHONF ([[4]](#4), [[5]](#5), [[6]](#6)).

Further work can include experimentation of enhancements to PaiNN with Stochastic Weight Averaging and/or layer optimization

## Milestones (time plan)
1. week 45: 
    - Project overview and dicussion of initial steps;
    - Literature review on provided resources about GNN and PaiNN;
    - Understand inputs and outputs of PaiNN;
    - Synopsis.
    
2. week 46: 
    - Preprocess the data to compute Adjacency matrix based on the atoms positions;
    - Write down the layers structure for: full architecture, message and update blocks;
    - Implement the basic PaiNN architecture that can handle the QM9 data.

3. week 47: 
    - Implement and test the layer structures;
    - Hyperparameter tunning and evaluation of performance.

4. week 48: 
    - Final PaiNN implementation;
    - Experiment with layer optimization.

5. week 49: 
    - Experiment with Stochastic Weight Averaging.
    - Define poster information and design.

6. week 50 and 51: Poster presentation and final review of report.

## References

<a id="1">[1]</a> Schütt, K., Unke, O., & Gastegger, M. (2021, July). Equivariant message passing for the prediction of tensorial properties and molecular spectra. In International Conference on Machine Learning (pp. 9377-9388). PMLR. \
<a id="2">[2]</a> Xu, M., Leskovec, J. (2023, March). Geometric Graph Learning From Representation to Generation. Guest lecture, Stanford CS224W Machine Learning with Graphs. \
<a id="3">[3]</a> Hamilton, W. L. (2020). Graph representation learning. Morgan & Claypool Publishers. Chapter 5 and 6. \
<a id="4">[4]</a> Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530. \
<a id="5">[5]</a> Ruddigkeit, L., Van Deursen, R., Blum, L. C., & Reymond, J. L. (2012). Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17. Journal of chemical information and modeling, 52(11), 2864-2875. \
<a id="6">[6]</a> Ramakrishnan, R., Dral, P. O., Rupp, M., & Von Lilienfeld, O. A. (2014). Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1), 1-7. \