# DGP-AMIO: Disease Gene Predictor based on Attention mechanism and integration of Multi-source gene Interaction networks and Omics
The code implementation of 'Integration of multi-source gene interaction networks and omics data with graph attention networks to identify novel disease genes' (paper can be accessed [here](https://www.biorxiv.org/content/10.1101/2023.12.03.569371v2)).
DGP-AMIO is a general disease gene predictor based on graph attention networks (GAT) which integrates gene interaction networks from multiple databases and multiomics.

## Dependencies
Our code environment is python 3.6 on Ubuntu 20.04 with CUDA 11.5. The packages of our environment which are dependencies for running DGP-AMIO are provided as follows:
* numpy==1.19.5
* pandas==1.15
* scikit-learn==0.24.2
* tqdm==4.63.1
* networkx==2.5.1
* torch==1.10.2+cu113
* torch_geometric==2.0.3

All of the packages can be installed through pip. Although not necessary, we strongly recommend GPU acceleration and conda for package management.

## Data Preparation
### Gene Interaction Networks
DGP-AMIO integrates different gene interaction networks, all of which can be downloaded from public databases. We provide preproccessed 10 gene interaction networks we used in the ```graph``` folder.
The preproccessing method is included in our paper.
### Omics and Known Disease Genes
Omics data as node (gene) features and known disease genes as labels are essential to train DGP-AMIO. We mainly used gene expression data from [GEO](https://www.ncbi.nlm.nih.gov/geo/). Other omics like DNA methylation and gene mutation 
can also be integrated as long as the data is provided. Known disease genes are collected from [Malacards](https://www.malacards.org/). Here we provide the preproccessed gene expression data and disease genes of asthma and Alzheimer
in the 'diseases' folder, so that you can directly run DGP-AMIO.

## Run DGP-AMIO 
If you want to run DGP-AMIO for disease gene prediction, you simply run:
```
python DGP-AMIO.py <disease name>
```
```DGP-AMIO.py``` does the following:
* Data preproccessing, and integrate gene interaction networks with omics and labels
* Train DGP-AMIO based on 5-fold cross validation and save the trained models
* Evaluate DGP-AMIO's performance on test set
* Load the trained models to predict on all unknown genes and save the results

```<disease name>``` is the disease you want to train DGP-AMIO and predict. Note that ```<disease name>``` needs to be consistent with the name of the folder that stores the corresponding disease data in the ```diseases``` directory to ensure proper data reading by the program.

## Train DGP-AMIO with Your Own Data
