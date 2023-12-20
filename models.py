import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv



class triGAT(nn.Module):
    def __init__(self, in_feats=1,
                 h_feats=16,
                 heads=8,
                 edge_feats=2,
                 dropout=0.2,
                 negative_slope=0.2,
                 **kwargs):
        super(triGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = Linear(3, 1)

        self.GAT11 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True, edge_dim=edge_feats)
        self.GAT12 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False, edge_dim=edge_feats)

        self.GAT21 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True, edge_dim=edge_feats)
        self.GAT22 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False, edge_dim=edge_feats)

        self.GAT31 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True)
        self.GAT32 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False)

    def forward(self, X, A1, A2, A3, edge_feature, return_alphas=False):
        #         alphas = []
        X1 = self.GAT11(X, A1, edge_feature)
        X1 = F.relu(X1)
        X1 = self.dropout(X1)

        X2 = self.GAT21(X, A2, edge_feature)
        X2 = F.relu(X2)
        X2 = self.dropout(X2)

        X3 = self.GAT31(X, A3)
        X3 = F.relu(X3)
        X3 = self.dropout(X3)

        X = torch.cat((X1, X2, X3), 1)

        X1 = self.GAT12(X, A1, edge_feature)
        X2 = self.GAT22(X, A2, edge_feature)
        X3 = self.GAT32(X, A3)

        X = torch.cat((X1, X2, X3), 1)
        X = self.ln(X)

        #         if return_alphas:
        #             X, alpha, edge_index = self.layers[-1](
        #                 X, A, return_alpha=True)
        #             alphas.append(alpha)
        #             return X, alphas, edge_index

        #         X = self.layers[-1](X, A)
        return X


class triGAT_without_edge_feature(nn.Module):
    def __init__(self, in_feats=1,
                 h_feats=16,
                 heads=8,
                 edge_feats=2,
                 dropout=0.2,
                 negative_slope=0.2,
                 **kwargs):
        super(triGAT_without_edge_feature, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = Linear(3, 1)

        self.GAT11 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True)
        self.GAT12 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False)

        self.GAT21 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True)
        self.GAT22 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False)

        self.GAT31 = GATConv(in_feats, h_feats, heads=heads, dropout=0.2, concat=True)
        self.GAT32 = GATConv(3 * h_feats * heads, 1, heads=1, dropout=0.2, concat=False)

    def forward(self, X, A1, A2, A3, edge_feature,return_alphas=False):
        #         alphas = []
        X1 = self.GAT11(X, A1)
        X1 = F.relu(X1)
        X1 = self.dropout(X1)

        X2 = self.GAT21(X, A2)
        X2 = F.relu(X2)
        X2 = self.dropout(X2)

        X3 = self.GAT31(X, A3)
        X3 = F.relu(X3)
        X3 = self.dropout(X3)

        X = torch.cat((X1, X2, X3), 1)

        X1 = self.GAT12(X, A1)
        X2 = self.GAT22(X, A2)
        X3 = self.GAT32(X, A3)

        X = torch.cat((X1, X2, X3), 1)
        X = self.ln(X)

        #         if return_alphas:
        #             X, alpha, edge_index = self.layers[-1](
        #                 X, A, return_alpha=True)
        #             alphas.append(alpha)
        #             return X, alphas, edge_index

        #         X = self.layers[-1](X, A)
        return X
