import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

import torch_geometric.nn as gnn


##############################################################################
#                               NeuralDevice
##############################################################################


class NeuralDevice(nn.Module):

    def __init__(self, eyes_dict, brain_nfs):
        
        super().__init__()

        self.eye_names = list(eyes_dict.keys())

        for eye_name, eye_hparams in eyes_dict.items():
            self.add_module(eye_name, NDEye(*eye_hparams))

        self.brain = NDBrain(*brain_nfs)

    def forward(self, x, batch_indices):

        eyes_outputs = []
        maxlen = max(batch_indices[0]) + 1
        for i, name in enumerate(self.eye_names):
            eye_output = self.get_submodule(name)(x[i], torch.tensor(batch_indices[i], device=device))
            L = eye_output.shape[0]
            if L < maxlen:
                nf = eye_output.shape[1]
                eye_output = torch.cat((eye_output, torch.zeros((maxlen - L, nf), device=device)), dim=0)
            eyes_outputs.append(eye_output)
        return self.brain(torch.cat(eyes_outputs, dim=1))


class NDEye(nn.Module):

    def __init__(self, r_in_nf, r_out_nf, c_out_nf):

        super().__init__()
        self.r_linear = nn.Linear(r_in_nf, r_out_nf) # receptor
        self.c_linear = nn.Linear(r_out_nf, c_out_nf) # collector

    def forward(self, x, batch_index):

        x = F.relu(self.r_linear(x))
        x = F.relu(self.c_linear(scatter(x, batch_index, dim=0, reduce='mean')))
        return x


class NDBrain(nn.Module):

    def __init__(self, in_nf, h_nf, out_nf):

        super().__init__()
        self.linear_1 = nn.Linear(in_nf, h_nf)
        self.linear_2 = nn.Linear(h_nf, out_nf) #, bias=False)

    def forward(self, x):

        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

##############################################################################
#                                 AtomMask
##############################################################################

class AtomMask(nn.Module):

    def __init__(self, atom, proba):

        super().__init__()
        self.register_buffer('atom', atom)
        self.proba = proba

    def forward(self, x):

        y = x.clone()
        mask = torch.rand(y.size(0)) < self.proba
        y[mask] = self.atom
        return y

##############################################################################
#                                   MPNN
##############################################################################


class Embed(nn.Module):

    def __init__(self,
                 node_in, node_h, node_out,
                 edge_in, edge_h, edge_out):
        
        super().__init__()
        self.node_2lp = nn.Sequential(
            nn.Linear(node_in, node_h),
            nn.ReLU(),
            nn.Linear(node_h, node_out),
            nn.LayerNorm(node_out)
        )
        self.edge_2lp = nn.Sequential(
            nn.Linear(edge_in, edge_h),
            nn.ReLU(),
            nn.Linear(edge_h, edge_out),
            nn.LayerNorm(edge_out)
        )

    def forward(self, x, edge_attr):

        return self.node_2lp(x), self.edge_2lp(edge_attr)


class MPNN(nn.Module):

    def __init__(self, atom_dim, bond_dim, atom_h_dim=16, bond_h_dim=8, out_dim=2, 
                 n_layers=4, rec=True, reg=False, default_atom=None, default_proba=None):
        super().__init__()

        self.reg = reg
        if self.reg:
            assert isinstance(default_atom, torch.tensor) and isinstance(default_proba, float)
            self.atom_mask = AtomMask(default_atom, default_proba)

        self.embed = Embed(
            node_in=atom_dim,
            node_h=atom_dim + h_dim,
            node_out=atom_h_dim,
            edge_in=bond_dim,
            edge_h=bond_dim + bond_h_dim,
            edge_out=bond_h_dim,
        )

        self.convolve = gnn.GINEConv(
            nn=nn.Sequential(
                nn.Linear(atom_h_dim, atom_h_dim*2),
                nn.ReLU(),
                nn.Linear(atom_h_dim*2, atom_h_dim)
            ),
            edge_dim=bond_h_dim,
        )

        self.n_layers = n_layers

        for i in range(self.n_layers):
            self.add_module(f'norm{i}', gnn.LayerNorm(atom_h_dim))

        self.gru = nn.GRU(
            input_size=atom_h_dim,
            hidden_size=atom_h_dim
        )

        self.updim = nn.Sequential(
            nn.Linear(atom_h_dim, atom_h_dim*2),
            nn.LeakyReLU(0.01),
            nn.Linear(atom_h_dim*2, atom_h_dim*2),
            nn.LayerNorm(atom_h_dim*2)
        )

        self.aggregate = gnn.aggr.MeanAggregation()

        self.clf = nn.Sequential(
            nn.Linear(atom_h_dim*2, atom_h_dim*4),
            nn.ReLU(),
            nn.Linear(atom_h_dim*4, out_dim, bias=False)
        )

        self.rec = rec
        self.reconstruct = nn.Sequential(
            nn.Linear(atom_h_dim*2, atom_h_dim*4),
            nn.ReLU(),
            nn.Linear(atom_h_dim*4, atom_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch_index=None):

        if self.reg:
            x = self.atom_mask(x)
        x, edge_attr = self.embed(x, edge_attr)
        h = x.unsqueeze(0)
        for i in range(self.n_layers):
            x = self.convolve(x, edge_index, edge_attr)
            x = self.get_submodule(f'norm{i}')(x).unsqueeze(0)
            x, h = self.gru(x, h)
            x = x.squeeze(0)
        u = self.updim(x)
        x = self.aggregate(u, index=batch_index)
        x = self.clf(x)
        if self.rec:
            r = self.reconstruct(u)
            return x, r
        return x


##############################################################################
#                                   RPET
##############################################################################


class RPETConv(gnn.MessagePassing):

    def __init__(self, dim):

        super().__init__(aggr='add')
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim)
        )
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def update(self, inputs, x):
        x = x_skip = self.norm1(x + inputs)
        x = self.ffn(x)
        x = self.norm2(x + x_skip)
        return x

    def message(self, x_i, x_j, edge_index, edge_attr):
        px_j = x_j + edge_attr
        q = self.q(x_i)
        k = self.k(px_j)
        v = self.v(px_j)
        a = (q * k).sum(1) * self.dim ** (-0.5)
        ae = torch.exp(a)
        aes = scatter(ae, index=edge_index[0], dim=0, reduce='sum')[edge_index[0]]
        s = ae / aes
        self.attn = s
        return s.view(-1, 1) * v


class RPETransformer(nn.Module):
    def __init__(self, in_nf, dim, n_layers, n_out=2, rec=True, reg=False, 
                 default_atom=None, default_proba=None):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers

        self.reg = reg
        if self.reg:
            assert isinstance(default_atom, torch.tensor) and isinstance(default_proba, float)
            self.atom_mask = AtomMask(default_atom, default_proba)

        self.atom_embedding = nn.Sequential(
            nn.Linear(in_nf, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
        self.conv = gnn.Sequential(
            'x, edge_index, edge_attr',
            [(RPETConv(dim), 'x, edge_index, edge_attr -> x',) for i in range(n_layers)]
        )
        self.updim = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.LeakyReLU(0.01),
            nn.Linear(dim*4, dim*4),
            nn.LayerNorm(dim*4),
        )
        self.readout = gnn.aggr.MeanAggregation()
        self.clf = nn.Sequential(
            nn.Linear(dim*4, n_out)
        )
        self.rec = rec
        self.reconstruct = nn.Sequential(
            nn.Linear(dim*4, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, in_nf)
        )

    def forward(self, x, edge_index, edge_attr, batch_index=None):
        if self.reg:
            x = self.atom_mask(x)
        x = self.atom_embedding(x)
        x = self.conv(x, edge_index, edge_attr)
        u = self.updim(x)
        x = self.readout(u, index=batch_index)
        x = self.clf(x)
        if self.rec:
            r = self.reconstruct(u)
            return x, r
        return x


##############################################################################
#                                   FFN
##############################################################################


class FFN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.out_dim)
        )
    def forward(self, x):
        return self.nn(x)
