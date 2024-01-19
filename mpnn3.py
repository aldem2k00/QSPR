import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data


class Featurizer:
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

    def __init__(self, molecules, other_threshold):
        n_mol = len(molecules)
        self.threshold = int(n_mol * other_threshold)
        f_count = self.__class__.feature_count(molecules)
        self.mappings = {}
        self.add_mappings(self.__class__.atom_feature_names, f_count)
        self.add_mappings(self.__class__.bond_feature_names, f_count)
        self.atom_dim = self.mappings[self.__class__.atom_feature_names[-1]]('default') + 1
        self.bond_dim = self.mappings[self.__class__.bond_feature_names[-1]]('default') + 1

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