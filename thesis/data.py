import requests
import io
import pandas as pd
from rdkit import Chem

import gzip


ESOL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
HIV_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
BBBP_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
TOX21_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

ESOL_Y = 'measured log solubility in mols per litre'
BBBP_Y = 'p_np'
HIV_Y = 'HIV_active'


def load_csv(url):
    res = requests.get(url)
    fp = io.StringIO(res.content.decode('utf-8'))
    df = pd.read_csv(fp)
    return df

def load_csv_gz(url):
    res = requests.get(url)
    fp = io.BytesIO(res.content)
    csv_text = gzip.decompress(fp.read()).decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def get_smiles_and_ys(df, smiles_col, ys_col):
    y_ser = df[ys_col]
    smiles_ser = df[smiles_col]
    mask = pd.notna(y_ser)
    y = y_ser[mask].to_numpy()
    smiles = smiles_ser[mask].to_list()
    assert len(y) == len(smiles)
    return smiles, y

def get_molecules(smiles, ys):
    ret_molecules = []
    ret_ys = []
    ret_smiles = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            ret_molecules.append(mol)
            ret_ys.append(ys[i])
            ret_smiles.append(s)
    return ret_molecules, ret_ys, ret_smiles


__all__ = ['ESOL_URL', 'HIV_URL', 'BBBP_URL', 'TOX21_URL', 'ESOL_Y', 'BBBP_Y',
           'HIV_Y', 'load_csv', 'load_csv_gz', 'get_smiles_and_ys', 
           'get_molecules']