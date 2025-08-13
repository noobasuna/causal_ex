import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math, os
import numpy as np
from pathlib import Path
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

class Causal_GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Causal_GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
#         adj_mat = np.load(mat_path)

#         self.register_buffer('adj', torch.from_numpy(adj_mat))

        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        
    def forward(self, input):
        b, n, c = input.shape
        #############################################################################
        
        #Implementation of Causal Discovery Algorithm in GCN. 
        #Project: Causal-Ex
        #Author: Pei-Sze, Tan ++
        
        #############################################################################
        
        #Transpose to get the desired shape for ca
        ex = input.cpu().detach().numpy()
        # print(input.shape)
        ex1 = np.transpose(ex, (1, 0, 2))
        data = np.reshape(ex1, (n, b * c))
        data = np.transpose(data, (1, 0))
        
        # Call the fci function and get the GeneralGraph object
        graph, edges = fci(data, fisherz, 0.01, verbose=False)
        #convert it into string
        G = graph.__str__()
        input_str = G
        # Split the input string into nodes and edges
        nodes_str, edges_str = input_str.split("\n\n")
        nodes = nodes_str.split(";")
        edges_list = edges_str.strip().split("\n")
        # Create a mapping of node labels to numeric indices
        label_to_index = {nodes[i]: i for i in range(len(nodes))}
        label_to_index.update(X1=0)
        # Convert edges to the desired format with numeric indices
        graph_edges = []
        for edge in edges_list:
            relation = edge.split()[1:]  # Extract the relation (e.g., ['X2', '-->', 'X1'])
            if '-->' in relation:  # Only include the components that involve '-->' (Directed Edge)
                start_node = label_to_index[relation[0]]
                end_node = label_to_index[relation[2]]
                graph_edges.append((start_node, end_node))

        # Build the adjacency matrix
        num_nodes = len(nodes)
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for edge in graph_edges:
            adj_matrix[edge[0]][edge[1]] = 1
        
        # Load the prior knowledge AU matrix based on ground truth labels
        co_matrix = np.load('./co_matrix.npy')
        # Perform addition function on both matrix
        adj_add = np.add(co_matrix, adj_matrix)
        # Normalize the matrix 
        norm = np.linalg.norm(adj_add, ord=1, axis=1, keepdims=True) + 1e-6
        add_mat = np.divide(adj_add, norm)
        adj = torch.from_numpy(add_mat)
        # self.register_buffer('adj', torch.from_numpy(add_mat))

        # Normalize the matrix
#         norm = np.linalg.norm(adj_matrix, ord=1, axis=1, keepdims=True) + 1e-6
#         adj_mat = np.divide(adj_matrix, norm)
        # adj = torch.from_numpy(adj_matrix)
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias1 = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias1', None)
        self.reset_parameters()
#         adj_mat = np.load(mat_path)

#         self.register_buffer('adj', torch.from_numpy(adj_mat))

        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias1 is not None:
            self.bias1.data.uniform_(-stdv, stdv)
        

    def forward(self, input):
        b, n, c = input.shape
        adj_m = np.load('./assets/cas(me)^2_ori.npy')
        # norm = np.linalg.norm(adj_m, ord=1, axis=1, keepdims=True) + 1e-6
        # adj_mat = np.divide(adj_m, norm)
        adj = torch.from_numpy(adj_m)
            # self.register_buffer('adj', torch.from_numpy(adj_mat))

        # Normalize the matrix
#         norm = np.linalg.norm(adj_matrix, ord=1, axis=1, keepdims=True) + 1e-6
#         adj_mat = np.divide(adj_matrix, norm)
        # adj = torch.from_numpy(adj_matrix)
        support = torch.bmm(input, self.weight1.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)
        #output = SparseMM(adj)(support)
        if self.bias1 is not None:
            return output + self.bias1
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.3):
        super(GCN, self).__init__()
        
        self.gc1 = Causal_GCN(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.bn2 = nn.BatchNorm1d(nout)
        self.dropout = dropout

    def forward(self, x):
        
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)
        
        # x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn2(x).transpose(1, 2).contiguous()
        x = F.relu(x)
        
        # x = self.gc2(x)
        # x = x.transpose(1, 2).contiguous()
        # x = self.bn2(x).transpose(1, 2).contiguous()
        # x = F.relu(x)
        # x = F.relu(self.gc2(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x
    
class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # print(mat_path)
        # mat_path = str(mat_path[0])
        # print(mat_path)
        # mat_path = Path(mat_path)
        # mat_path = './assets/cas(me)^2.npy'
        # mat_path = list(mat_path)
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16))
        #self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path))
        in_dim = 192#24

        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=2),

            # # receptive filed: 7
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=2),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
                            bias=False), 
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=2),
            # torch.nn.AdaptiveAvgPool1d(270)
            
            # torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
            #                 bias=False), 
            # torch.nn.BatchNorm1d(64),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=2, stride=1, padding=1, dilation=2),
        )
        # 0:micro(start,end,None),    3:macro(start,end,None),
        # 6:micro_apex,7:macro_apex,  8:micro_action, macro_action
        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        # print(vid_name)
        x = x.reshape(b*t, n, c)  # (b*t, n, c)
        # print('b, t, n, c:', b, t, n, c)
        x = self.graph_embedding(x).reshape(b, t, -1).transpose(1, 2)   # (b, C=384=12*32, t)
        # print(x.shape)
        #x = self.graph_embedding(x).reshape(b, t, n, 16)
        x = self._sequential(x)
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    import yaml
    # load config & params.
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    
    x = torch.randn((16, 64, 12, 2))        # (b, t, n, c)
    model = PEM(opt)
    
    out = model(x)
    print(out.shape)
    