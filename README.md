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
can also be integrated as long as the data is provided. Known disease genes are collected from [Malacards](https://www.malacards.org/). Here we provide the preproccessed gene expression data ([GSE143303](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143303) and [GSE184942](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184942)) and disease genes of asthma and Alzheimer
in the ```diseases``` folder, so that you can directly run DGP-AMIO.

## Run DGP-AMIO 
If you want to run DGP-AMIO for disease gene prediction, simply run:
```
python DGP-AMIO.py <disease name>
```
For example, ``` python DGP-AMIO.py asthma```

```DGP-AMIO.py``` does the following:
* Data preproccessing, and integrate gene interaction networks with omics and labels
* Train DGP-AMIO based on 5-fold cross validation and save the trained models
* Evaluate DGP-AMIO's performance on test set
* Load the trained models to predict on all unknown genes and save the results

```<disease name>``` is the disease you want to train DGP-AMIO and predict. Note that ```<disease name>``` needs to be consistent with the name of the folder that stores the corresponding disease data in the ```diseases``` directory to ensure proper data reading by the program.

## Train DGP-AMIO with Your Own Data
* If you want to integrate other gene interaction networks, please organize them in the form of edge lists with a size of n×2, where n is the number of directed edges (following the format of the provided gene network files), and save the edge lists in CSV files named ```edges_<database name>.csv``` and place them in the ```graphs``` folder. After this, please don't forget to include ```<database name>``` in the list variable ```databases``` in ```DGP-AMIO.py```
* If you want to use your own expression data, please organize the data in a CSV file named ```<disease name>_expression.csv``` where the first column is the gene names and place it in the folder of the corresponding disease. The code can run without any modification. If you want to use other omics or integrate multiomics simultaneously, you need to replace the expression data with your own data and make necessary modifications to the data reading part of ```DGP-AMIO.py``` to ensure its proper execution
* If you want to use other labeled data, just replace the ```<disease name>_gene.csv``` with your own labeled data
* If you want to train DGP-AMIO and make predictions of a new disease, you need to create ```diseases/<disease name>``` folder and place the prepared omics data and known disease genes in it
* Note that when preparing your own data, please represent genes in all your data files with gene symbols.  
