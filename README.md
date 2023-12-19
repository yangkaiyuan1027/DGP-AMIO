# DGP-AMIO: Disease Gene Predictor based on Attention mechanism and integration of Multi-source gene Interaction networks and Omics
The code implementation of 'Integration of multi-source gene interaction networks and omics data with graph attention networks to identify novel disease genes' (paper can be accessed [here](https://www.biorxiv.org/content/10.1101/2023.12.03.569371v2)).
DGP-AMIO is a general disease gene predictor which integrates gene interaction networks from multiple databases and multiomics.
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
