import rdkit
from rdkit import Chem

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_scatter import scatter

import matplotlib.pyplot as plt

import lipo
from mpnn3 import Featurizer

featurizer = Featurizer(lipo.molecules, other_threshold=0.02)
mol_data_list = featurizer.featurize(lipo.molecules)

def random_split(data_list, ys, test_ratio):
    """given list of torch_geometric.data.Data (data_list) and respective 
    targets (ys), splits them into training and test sublists randomly"""
    test_len = int(len(data_list) * test_ratio)
    indices = np.random.permutation(np.arange(len(data_list)))
    test_indices, train_indices = indices[:test_len], indices[test_len:]
    test_list = [data_list[i] for i in test_indices]
    test_ys = [ys[i] for i in test_indices]
    train_list = [data_list[i] for i in train_indices]
    train_ys = [ys[i] for i in train_indices]
    return train_list, train_ys, test_list, test_ys

train_list, train_ys, test_list, test_ys = random_split(mol_data_list, lipo.values, 0.2)

def load_batches(data_list, ys, batch_size, shuffle=True):
    assert len(data_list) == len(ys)
    indices = np.arange(len(data_list))
    if shuffle:
        indices = np.random.permutation(indices)
    for batch_index in range((len(data_list)-1) // batch_size + 1):
        start = batch_size * batch_index
        stop = start + batch_size
        sublist = [data_list[i] for i in indices[start:stop]]
        X = Batch.from_data_list(sublist)
        y = torch.tensor([ys[i] for i in indices[start:stop]])
        yield X, y

class MPNN(MessagePassing):
    def __init__(
            self, 
            n_node_in_features, 
            n_node_out_features,
            n_message_features,
            n_edge_features):
        super().__init__(aggr='add')
        self.message_lin = nn.Linear(
            2 * n_node_in_features + n_edge_features,
            n_message_features
            )
        self.message_norm = nn.LayerNorm(n_message_features)
        self.update_lin = nn.Linear(
            n_message_features + n_node_in_features,
            n_node_out_features
            )
        self.update_norm = nn.LayerNorm(n_node_out_features)
        self.tanh = torch.tanh

    def forward(self, x, edge_index, edge_attr):

        return (self.propagate(edge_index, x=x, edge_attr=edge_attr),
                edge_index, edge_attr)
    def update(self, inputs, x):
        return self.tanh(
            self.update_norm(
                self.update_lin(
                    torch.cat((inputs, x), dim=1)
                )
            )
        )
    def message(self, x_j, x_i, edge_index, edge_attr):
        #x_j and x_i shapes: (E, node_dim)
        #edge_attr shape: (E, edge_dim)
        return (
            self.tanh(
                self.message_norm(
                    self.message_lin(
                        torch.cat(
                            (x_j, x_i, edge_attr),
                             dim=1
                        )
                    )
                )
            )
        )

class Model(nn.Module):
    def __init__(
        self,
        n_atom_features,
        n_bond_features,
        mpnn_layers_args,
        hidden_layer_size,
        out_features
    ):
        super().__init__()
        self.atoms_linear = nn.Linear(n_atom_features, mpnn_layers_args[0][0])
        self.bonds_linear = nn.Linear(n_bond_features, mpnn_layers_args[0][3])
        self.mpnns = SeqMult(*(MPNN(*args) for args in mpnn_layers_args))
        self.fc = nn.Sequential(
            nn.Linear(mpnn_layers_args[-1][1], hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, out_features)
        )
    def forward(self, batch):
        x = self.atoms_linear(batch.x)
        edge_index = batch.edge_index
        edge_attr = self.bonds_linear(batch.edge_attr)
        x, _, _ = self.mpnns(x, edge_index, edge_attr)
        x = self.fc(x)
        return scatter(x.t(), batch.batch, reduce='mean')

class SeqMult(nn.Module):
    """almost nn.Sequential, but modules can take multiple arguments as input"""
    def __init__(self, *submodules):
        super().__init__()
        self.submodules = []
        for i, module in enumerate(submodules):
            self.add_module('module'+str(i), module)
            self.submodules.append(module)
    def forward(self, *args):
        x = args
        for module in self.submodules:
            x = module(*x)
        return x

model = Model(
    n_atom_features=28, 
    n_bond_features=10, 
    mpnn_layers_args=(
        ((8, 16, 16, 4),)
        + ((16, 16, 16, 4),) * 10
        + ((16, 128, 64, 4),)
    ), 
    hidden_layer_size=64, 
    out_features=1
)

loss_func = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=0.002)
n_epoch = 50
for i in range(n_epoch):
    avg_loss = 0
    for X, y in load_batches(train_list, train_ys, 32):
        pred = model(X)[0]
        loss = loss_func(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        avg_loss += loss.item() * y.size()[0]
    print('epoch:', str(i+1) + ',\taverage loss:', round(avg_loss / len(train_list), 3))