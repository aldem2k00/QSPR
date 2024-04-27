import torch
from rdkit import Chem
import networkx as nx
import itertools as it


class NDFeaturizer:
    atom_feature_names = (
        'number',
        'degree',
        'charge',
        'num_h',
        'hybrid',
        'aromatic'
    )

    bond_feature_names = (
        'bondtype',
        'conjug',
        'isinring'
    )

    atom_methods = (
        Chem.Atom.GetAtomicNum,
        Chem.Atom.GetDegree,
        Chem.Atom.GetFormalCharge,
        Chem.Atom.GetTotalNumHs,
        Chem.Atom.GetHybridization,
        Chem.Atom.GetIsAromatic
    )

    bond_methods = (
        Chem.Bond.GetBondType,
        Chem.Bond.GetIsConjugated,
        Chem.Bond.IsInRing
    )

    def __init__(self, molecules, other_threshold, n_eyes=2):
        n_mol = len(molecules)
        self.threshold = int(n_mol * other_threshold)
        f_count = self.__class__.feature_count(molecules)
        self.mappings = {}
        self.add_mappings(self.__class__.atom_feature_names, f_count)
        self.add_mappings(self.__class__.bond_feature_names, f_count)
        self.atom_dim = self.mappings[self.__class__.atom_feature_names[-1]]('default') + 1
        self.bond_dim = self.mappings[self.__class__.bond_feature_names[-1]]('default') + 1
        self.n_eyes = n_eyes

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

    def atom_tensor(self, atom):
        ret = torch.zeros(self.atom_dim)
        for key, method in zip(
            self.__class__.atom_feature_names,
            self.__class__.atom_methods
        ):
            ret[self.mappings[key](method(atom))] = 1.0
        return ret

    def bond_tensor(self, bond):
        ret = torch.zeros(self.bond_dim)
        for key, method in zip(
            self.__class__.bond_feature_names,
            self.__class__.bond_methods
        ):
            ret[self.mappings[key](method(bond))] = 1.0
        return ret

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
        # print(a)
        ret = [a,]
        if self.n_eyes > 1:
            cp_idxs = self.__class__.get_connected_pairs(g)
            cp_list = []
            for cp_idx in cp_idxs:
                cp_list.append(a[cp_idx].view(1, -1))
                cp_list.append(a[cp_idx[::-1]].view(1, -1))
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

    @classmethod
    def feature_count(cls, molecules):
        ret = {name: {} for name in cls.atom_feature_names + cls.bond_feature_names}
        for mol in molecules:
            features = cls.mol_features(mol)
            for key, val in features.items():
                for x in val:
                    if x not in ret[key]:
                        ret[key][x] = 1
                    else:
                        ret[key][x] += 1
        for name in ret:
            ret[name] = dict(sorted(ret[name].items()))
        return ret

    @classmethod
    def mol_features(cls, mol):
        ret = {name: [] for name in cls.atom_feature_names + cls.bond_feature_names}
        for atom in mol.GetAtoms():
            for key, method in zip(
                cls.atom_feature_names, cls.atom_methods
            ):
                val = method(atom)
                if val not in ret[key]:
                    ret[key].append(val)
        for bond in mol.GetBonds():
            for key, method in zip(
                cls.bond_feature_names, cls.bond_methods
            ):
                val = method(bond)
                if val not in ret[key]:
                    ret[key].append(val)
        return ret

    @staticmethod
    def make_mapping(d, default=0):
        dcopy = d.copy()
        return lambda x: dcopy[x] if x in dcopy.keys() else default
