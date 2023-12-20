import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from models import triGAT
from train_test import evalAUC,model_train,test
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, PrecisionRecallDisplay, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import scanpy as sc

disease = '/'+sys.argv[1]

def data_process():
    databases=['RegNetwork','trrust','JASPAR','MOTIFMAP','EVEX','CHEA','string','CPDB','IREF']
    edges = pd.read_csv('./graphs/edges_kegg.csv')
    edges = edges.dropna()
    edges = edges.drop_duplicates()
    edges['kegg'] = np.ones(len(edges.index))
    for name in databases:
        graph = pd.read_csv('./graphs/edges_'+name+'.csv')
        graph = graph.dropna()
        graph = graph.drop_duplicates()  # remove the repeated edges
        graph[name] = np.ones(len(graph.index))
        edges = pd.merge(edges, graph, on=['source', 'target'], how='outer')
    edges.fillna(0, inplace=True)

    print("Number of edges before reprocessing:",len(edges.values))

    genes = np.union1d(edges.values[:,0],edges.values[:,1])
    print("Number of genes（nodes）in the graph:",len(genes))

    # expression data from GEO
    expression = pd.read_csv('diseases'+disease+disease+'_expression.csv',index_col=0)
    expression = expression.groupby('Gene').mean()
    expression = expression[expression.apply(np.sum, axis=1) != 0]  # drop the zero-expression data
    # print(expression.shape)

    X = np.zeros((len(genes), 0))
    X = pd.DataFrame(X, index=genes)
    columns = [f'expression_{i}' for i in range(expression.shape[1])]
    expression.columns = columns
    X = X.join(expression, how="left")
    X.dropna(inplace=True)  # graph overlap with expression data

    genes_drop = np.setdiff1d(genes, X.index.values)

    # edges overlap with expression
    genes_drop = pd.DataFrame(genes_drop, columns=['source'])
    genes_drop['label1'] = np.ones(len(genes_drop.index))
    edges = pd.merge(edges, genes_drop, on=['source'], how='left')
    genes_drop.columns = ['target', 'label2']
    edges = pd.merge(edges, genes_drop, on=['target'], how='left')
    edges.fillna(0, inplace=True)
    edges['label'] = edges['label1']+ edges['label2']
    edges.drop(columns=['label1', 'label2'], inplace=True)
    edges = edges[edges['label'] == 0]
    edges = edges.values[:, :-1]
    print("Number of edges after overlapping with expression:", len(edges))

    genes=np.union1d(edges[:,0],edges[:,1])
    print("Number of genes（nodes）in the graph after overlapping with expression data:",len(genes))
    X = X.loc[genes]

    N = len(X)
    mapping = dict(zip(genes, range(N)))
    # print("Number of edges before removing self loops",len(edges))

    # Remove self loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    print("Number of edges after removing self loops",len(edges))

    edge_index = edges[:, :2]
    edge_feature = edges[:, 2:]
    edge_feature = edge_feature.astype(float)
    edge_index = np.vectorize(mapping.__getitem__)(edge_index)

    degrees = np.zeros((N, 1))
    nodes, counts = np.unique(edge_index, return_counts=True)
    degrees[nodes, 0] = counts
    X = X.values

    X = np.concatenate([X, degrees.reshape((-1, 1))], 1)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    print("Shape of node features", X.shape)

    # Torch -------------------------------------------------
    edge_index = torch.from_numpy(edge_index.T)
    edge_index = edge_index.to(torch.long).contiguous()
    X = torch.from_numpy(X).to(torch.float32)
    edge_feature = torch.from_numpy(edge_feature).to(torch.float32)

    edge_new1 = torch.empty(edge_index.shape)
    edge_new1 = edge_new1.to(torch.long)
    edge_new1[[0, 1], :] = edge_index[[1, 0], :]

    edge_new2 = torch.cat((edge_index,edge_new1), dim=1)
    edge_new2 = torch.unique(edge_new2, dim=1)

    return genes, mapping, X, edge_index, edge_new1, edge_new2, edge_feature

def data_split(genes):
    # train and test split
    label_gene = pd.read_csv('diseases'+disease+disease+'_gene.csv',index_col=0)

    # filter the labeled genes not in the graph
    label_gene = label_gene.loc[np.intersect1d(label_gene.index, genes)]
    known_target_genes = label_gene.index.values

    zero_gene = np.setdiff1d(genes, label_gene.index)
    random.seed(1)
    sample_zero_gene = random.sample(list(zero_gene), len(label_gene.index))
    for gene in sample_zero_gene:
        new_line = pd.DataFrame({'Label': 0}, index=[gene])
        label_gene = label_gene.append(new_line)

    # 5-fold cross validation
    train, test = train_test_split(label_gene, test_size=0.2, stratify=label_gene, random_state=0)
    test_idx = [mapping[t] for t in test.index]
    test_y = torch.tensor(test.Label.astype(int), dtype=torch.float32)

    data_blocks = []
    idx_y = []
    for i in range(4):
        train, block = train_test_split(train, test_size=1 / (5 - i), stratify=train, random_state=0)
        data_blocks.append(block)
    data_blocks.append(train)
    for i in range(5):
        val = data_blocks[0]
        del data_blocks[0]
        train = pd.concat(data_blocks, axis=0)
        train_idx = [mapping[t] for t in train.index]
        train_y = torch.tensor(train.Label.astype(int), dtype=torch.float32)
        val_idx = [mapping[t] for t in val.index]
        val_y = torch.tensor(val.Label.astype(int), dtype=torch.float32)
        data_blocks.append(val)
        idx_y.append((train_idx, train_y, val_idx, val_y))

    return idx_y, test_idx, test_y, known_target_genes


if __name__ == '__main__':
    genes, mapping, X, edge_index, edge_new1, edge_new2, edge_feature = data_process()
    idx_y, test_idx, test_y, known_target_genes = data_split(genes)
    params = {
        'lr': 0.005,  # 0.005
        'weight_decay': 5e-4,  #5e-4
        'h_feats': 16,  # 16
        'heads': 8,  # 8
        'dropout': 0.2,  # 0.3
        'negative_slope': 0.2}
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train
    for i in range(5):  # 5-fold cross validation
        train_idx, train_y, val_idx, val_y = idx_y[i]
        model = model_train(params, X, edge_index, edge_new1, edge_new2, edge_feature, train_y, train_idx, val_y,
                            val_idx, savepath='models/model' + str(i + 1) + '.pt',DEVICE=DEVICE)

    # Test
    AUCs = []
    AUPRs = []
    X = X.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_new1 = edge_new1.to(DEVICE)
    edge_new2 = edge_new2.to(DEVICE)
    edge_feature = edge_feature.to(DEVICE)
    test_y = test_y.to(DEVICE)
    preds_p_mean = 0

    for i in range(5):
        model = triGAT(in_feats=X.shape[1], edge_feats=edge_feature.shape[1], **params)
        model.load_state_dict(torch.load("./models/model" + str(i + 1) + ".pt"))
        model.to(DEVICE).eval()
        with torch.no_grad():
            logits = model(X, edge_index, edge_new1, edge_new2, edge_feature)
            preds_p = torch.sigmoid(logits).to(torch.float32)[:, 0]
            preds_p_mean = preds_p_mean + preds_p / 5
            auc = evalAUC(model, X, edge_index, edge_new1, edge_new2, edge_feature, test_y, test_idx)
            AUCs.append(auc)
            aupr = average_precision_score(test_y.cpu(), preds_p[test_idx].cpu())
            AUPRs.append(aupr)
    preds = (preds_p_mean[test_idx] > 0.5).to(torch.float32)
    # print("5-fold AUCs:", AUCs)
    # print("5-fold AUPRs:", AUPRs)
    fpr, tpr, threshold = metrics.roc_curve(test_y.cpu(), preds_p_mean[test_idx].cpu())
    roc_auc = metrics.auc(fpr, tpr)
    AUPR = average_precision_score(test_y.cpu(), preds_p_mean[test_idx].cpu())

    # print("precision:", precision_score(test_y.cpu(), preds.cpu()))
    print("AUROC:", roc_auc)
    print("AUPR:", AUPR)

    # Save results
    df = pd.DataFrame(columns=['Gene', 'Pred_p'])
    df['Gene'] = genes
    df['Pred_p'] = preds_p_mean.cpu()
    df.set_index('Gene', inplace=True)
    df.drop(index=known_target_genes, inplace=True)
    df.sort_values(by='Pred_p', inplace=True, ascending=False)
    # df.to_csv('results'+disease+'.csv')

    # plt.figure(figsize=(4,4))
    # plt.title('Test ROC')
    # plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.3f' %roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    #plt.savefig('results'+disease+'_ROCAUC.png')

    # precision, recall, _ = precision_recall_curve(test_y.cpu(), preds_p_mean[test_idx].cpu())
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot(label='Test AUPR=%0.3f' %AUPR)
    # fig = plt.gcf()
    #fig.savefig('results'+disease+'_AUPR.png')

