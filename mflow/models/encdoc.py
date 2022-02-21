import torch
from torch import nn
import numpy as np
from scipy import linalg as la
from mflow.models.layers import GraphConvolution, MultiDenseLayer, GraphAggregation

# define encoder : rgcn + mlp --> n*n*c + n*d
# define decode : mlp --> edges_logits, nodes_logits
# conv_dim, linear_dim
# [256], [512, 256]
# [64], [128, 64]


class graphEncoder(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, linear_dim, vertexes, edges, nodes, with_features=False, f_dim=0, dropout_rate=0.):
        super(graphEncoder, self).__init__()
        # gcn时graph channel 为bond type， 生成新graph时channel为bond+1
        # b_dim = edges - 1
        graph_conv_dim, aux_dim = conv_dim
        self.activation_f = torch.nn.Tanh()
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f)
        self.norm1 = nn.BatchNorm1d(aux_dim) ###
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh())
#         self.norm2 = nn.BatchNorm1d(linear_dim[-1]) ###

        self.emb_mean = nn.Linear(linear_dim[-1], edges*vertexes*vertexes+vertexes*nodes)
        self.emb_logvar = nn.Linear(linear_dim[-1], edges*vertexes*vertexes+vertexes*nodes)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, adj, node):
        adj = adj[:, :-1, :, :]
        output = self.gcn_layer(node, adj)
        output = self.agg_layer(output, node)
#         output = self.norm1(output)
        output = self.multi_dense_layer(output)
        h_mu = self.emb_mean(output)
        h_logvar = self.emb_logvar(output)
        h = self.reparameterize(h_mu, h_logvar)      
        return h, h_mu, h_logvar
    
    
    def embed(self, adj, node, hidden=None, activation=None):
        adj = adj[:, :-1, :, :]
        h_node = self.gcn_layer(node, adj)  
#         h = self.agg_layer(h_node, node, hidden)
#         h = self.multi_dense_layer(h)
        
        return h_node.detach()   