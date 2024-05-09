import rdkit
from rdkit import Chem

import numpy as np
import torch

from torch_geometric.data import Data

import networkx as nx

import itertools as it


class Featurizer:

    atom_feature_names = (
        'number',
        'degree',
        'charge',
        'num_h',
        'hybrid',
        'aromatic',
        'atom_isinring'
    )

    bond_feature_names = (
        'bondtype',
        'conjug',
        'bond_isinring'
    )

    atom_methods = dict(
        zip(
            atom_feature_names, (
                Chem.Atom.GetAtomicNum,
                Chem.Atom.GetDegree,
                Chem.Atom.GetFormalCharge,
                Chem.Atom.GetTotalNumHs,
                Chem.Atom.GetHybridization,
                Chem.Atom.GetIsAromatic,
                Chem.Atom.IsInRing
            )
        )
    )

    bond_methods = dict(
        zip(
            bond_feature_names, (
                Chem.Bond.GetBondType,
                Chem.Bond.GetIsConjugated,
                Chem.Bond.IsInRing
            )
        )
    )

    def __init__(self, molecules=None, mappings_data=None, 
                 other_threshold=0.0, atom_features='all', bond_features='all'):
        
        if mappings_data is not None:

            self.atom_feature_names = []
            self.atom_methods = {}
            self.bond_feature_names = []
            self.bond_methods = {}
            for name in mappings_data.keys():
                if name in self.__class__.atom_feature_names:
                    self.atom_feature_names.append(name)
                    self.atom_methods[name] = self.__class__.atom_methods[name]
                elif name in self.__class__.bond_feature_names:
                    self.bond_feature_names.append(name)
                    self.bond_methods[name] = self.__class__.bond_methods[name]
                else:
                    raise ValueError('not supported feature')

            self.mappings = {}
            for name, d in mappings_data.items():
                dcopy = d.copy()
                default_index = dcopy.pop('Default')
                self.mappings[name] = self.__class__.make_mapping(dcopy, default_index)
            self.mappings_data = mappings_data

        elif molecules is not None:

            n_mol = len(molecules)
            self.threshold = int(n_mol * other_threshold)
            f_count = self.feature_count(molecules)
            self.mappings = {}
            self.mappings_data = {}
            self.add_mappings(self.atom_feature_names, f_count)
            self.add_mappings(self.bond_feature_names, f_count)
            if atom_features == 'all':
                self.atom_feature_names = self.__class__.atom_feature_names
                self.atom_methods = self.__class__.atom_methods
            else:   
                assert isinstance(atom_features, tuple) or isinstance(atom_features, list)
                assert len(atom_features)
                self.atom_methods = {}
                for name in atom_features:
                    self.atom_methods[name] = self.__class__.atom_methods[name]
                self.atom_feature_names = atom_features

            if bond_features == 'all':
                self.bond_feature_names = self.__class__.bond_feature_names
                self.bond_methods = self.__class__.bond_methods
            elif bond_features is None:
                self.bond_feature_names = ()
                self.bond_methods = {}
            else:
                assert isinstance(bond_features, tuple) or isinstance(bond_features, list)
                assert len(bond_features)
                self.bond_methods = {}
                for name in bond_features:
                    self.bond_methods[name] = self.__class__.bond_methods[name]
                self.bond_feature_names = bond_features

        else:
            raise TypeError('either "molecules" or "mappings_data" must be provided as a keyword argument')

        self.atom_dim = self.mappings[self.atom_feature_names[-1]]('default') + 1
        if self.bond_feature_names:
            self.bond_dim = self.mappings[self.bond_feature_names[-1]]('default') + 1
        else:
            self.bond_dim = 0

    def add_mappings(self, names, f_count):
        i = -1
        for name in names:
            i += 1
            d = f_count[name]
            filtered = {}
            for key, val in d.items():
                if val >= self.threshold:
                    filtered[key] = i
                    i += 1
            self.mappings[name] = self.__class__.make_mapping(filtered, i)
            filtered['Default'] = i
            self.mappings_data[name] = filtered

    def atom_tensor(self, atom):
        ret = torch.zeros(self.atom_dim)
        for name in self.atom_feature_names:
            method = self.atom_methods[name]
            ret[self.mappings[name](method(atom))] = 1.0
        return ret

    def bond_tensor(self, bond):
        ret = torch.zeros(self.bond_dim)
        for name in self.bond_feature_names:
            method = self.bond_methods[name]
            ret[self.mappings[name](method(bond))] = 1.0
        return ret

    def feature_count(self, molecules):
        ret = {name: {} for name in self.atom_feature_names + self.bond_feature_names}
        for mol in molecules:
            features = self.mol_features(mol)
            for key, val in features.items():
                for x in val:
                    if x not in ret[key]:
                        ret[key][x] = 1
                    else:
                        ret[key][x] += 1
        for name in ret:
            ret[name] = dict(sorted(ret[name].items()))
        return ret

    def mol_features(self, mol):
        ret = {name: [] for name in self.atom_feature_names + self.bond_feature_names}
        for atom in mol.GetAtoms():
            for name in self.atom_feature_names:
                method = self.atom_methods[name]
                val = method(atom)
                if val not in ret[name]:
                    ret[name].append(val)

        if not self.bond_feature_names:
            return ret

        for bond in mol.GetBonds():
            for name in self.bond_feature_names:
                method = self.bond_methods[name]
                val = method(bond)
                if val not in ret[name]:
                    ret[name].append(val)

        return ret

    @staticmethod
    def make_mapping(d, default=0):
        dcopy = d.copy()
        return lambda x: dcopy[x] if x in dcopy.keys() else default



class NDFeaturizer(Featurizer):

    def __init__(self, n_eyes=3, **kwargs):
        super().__init__(bond_features=None, **kwargs)
        self.n_eyes = n_eyes

    def featurize(self, molecules):
        featurized = []
        for mol in molecules:
            featurized.append(self.mol_to_nd_input(mol))
        return featurized

    def mol_to_nd_input(self, mol):
        g = nx.Graph()
        a_list = []
        for i, atom in enumerate(mol.GetAtoms()):
            g.add_node(i)
            a_list.append(self.atom_tensor(atom).view(1, -1))
        for bond in mol.GetBonds():
            g.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx())
        a = torch.cat(a_list, dim=0)
        ret = [a,]
        if self.n_eyes > 1:
            cp_idxs = self.__class__.get_connected_pairs(g)
            cp_list = []
            for cp_idx in cp_idxs:
                cp_list.append(a[cp_idx].view(1, -1))
                cp_list.append(a[cp_idx[::-1]].view(1, -1))
            if len(cp_list) == 0:
                cp = torch.empty((0, self.atom_dim*2))
            else:
                cp = torch.cat(cp_list, dim=0)
            ret.append(cp)
        if self.n_eyes > 2:
            ct_idxs = self.__class__.get_connected_triads(g)
            ct_list = []
            for ct_idx in ct_idxs:
                for perm in it.permutations(ct_idx, 3):
                    ct_list.append(a[ct_idx].view(1, -1))
            if len(ct_list) == 0:
                ct = torch.empty((0, self.atom_dim*3))
            else:
                ct = torch.cat(ct_list, dim=0)
            ret.append(ct)
        return tuple(ret)

    @classmethod
    def get_connected_pairs(cls, g):
        ret = []
        for node in g.nodes():
            for neighbor in g.neighbors(node):
                if node < neighbor:
                    ret.append([node, neighbor])
        return ret

    @classmethod
    def get_connected_triads(cls, g):
        ret = []
        for node in g.nodes():
            neighbors = [x for x in g.neighbors(node)]
            if len(neighbors) < 2:
                continue
            for pair in it.combinations(neighbors, 2):
                ret.append([pair[0], node, pair[1]])
        return ret


class MPNNFeaturizer(Featurizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def featurize(self, molecules):
        featurized = []
        for mol in molecules:
            featurized.append(self.mol_to_pyg_data(mol))
        return featurized

    def mol_to_pyg_data(self, mol):
        x = torch.zeros(mol.GetNumAtoms(), self.atom_dim)
        edge_attr = torch.zeros(mol.GetNumBonds(), self.bond_dim)
        edge_index = torch.zeros(2, mol.GetNumBonds(), dtype=torch.long)
        for i, atom in enumerate(mol.GetAtoms()):
            x[i] = self.atom_tensor(atom)
        for i, bond in enumerate(mol.GetBonds()):
            edge_index[:,i] = torch.tensor([
                bond.GetBeginAtomIdx(), 
                bond.GetEndAtomIdx()])
            edge_attr[i] = self.bond_tensor(bond)
        edge_index = torch.cat((edge_index, edge_index.flip(dims=(0,))), dim=1).contiguous()
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0).contiguous()
        return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


class RPETFeaturizer(Featurizer):

    def __init__(self, max_distance=8, pe_dim=16, **kwargs):
        super().__init__(bond_features=None, **kwargs)
        self.max_distance = max_distance
        self.penc = PositionalEncoding(pe_dim, max_len=max_distance+1)

    def featurize(self, molecules):
        featurized = []
        for mol in molecules:
            featurized.append(self.mol_to_rpet_input(mol))
        return featurized

    def mol_to_rpet_input(self, mol):
        g = nx.Graph()
        a_list = []
        for i, atom in enumerate(mol.GetAtoms()):
            g.add_node(i)
            a_list.append(self.atom_tensor(atom).view(1, -1))
        for bond in mol.GetBonds():
            g.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx())
        x = torch.cat(a_list, dim=0)
        rp = self.__class__.graph_rp(g)
        mask = rp <= self.max_distance
        rp[~mask] = 0
        edge_index = torch.tensor(np.stack(np.where(rp > 0)))
        edge_attr = self.penc(torch.tensor(rp[rp > 0] - 1)).squeeze(1)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def graph_rp(cls, g):
        spl = list(nx.all_pairs_shortest_path_length(g))
        N = g.number_of_nodes()
        ret = np.zeros((N, N), dtype=np.int32)
        for x in spl:
            for y in x[1].keys():
                ret[x[0], y] = x[1][y]
        return ret