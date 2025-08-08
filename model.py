#!/usr/bin/env python
"""
DGCNN using DynamicEdgeConv from PyGeometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool
from torch_geometric.nn.pool import knn_graph

class DGCNN_PyG(nn.Module):

    def __init__(self, args, output_channels=40):
        super().__init__()
        self.dropout = args.dropout
        self.emb_dims = args.emb_dims
        self.k = args.k

        def mlp(in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(in_channels,out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2)
            )

        self.conv1 = EdgeConv(mlp(6,64), aggr='max')
        self.conv2 = EdgeConv(mlp(128,64), aggr='max')
        self.conv3 = EdgeConv(mlp(128,128), aggr='max')
        self.conv4 = EdgeConv(mlp(256, 256), aggr='max')

        self.linear1 = nn.Linear(512, self.emb_dims)
        self.bn1 = nn.BatchNorm1d(self.emb_dims)

        self.linear2 = nn.Linear(self.emb_dims*2, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(self.dropout)
        self.linear3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(self.dropout)
        self.linear4 = nn.Linear(256, output_channels)

    def forward(self, data):
        x, batch = data.pos, data.batch

        edge_index = knn_graph(x, k = self.k, batch=batch, loop=False)

        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)

        x_cat = torch.cat([x1, x2, x3, x4], dim = 1)
        x = F.leaky_relu(self.bn1(self.linear1(x_cat)), negative_slope=0.2)

        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = torch.cat((x_max, x_mean), dim = 1)

        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope = 0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn3(self.linear3(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear4(x) # batch size, num_classes
