from rdkit import Chem

path_to_csv = 'Lipophilicity.csv'
with open(path_to_csv, 'r') as fp:
    lines = fp.readlines()
values = []
smiles = []
for line in lines[1:]:
    content = line.split(',')
    values.append(float(content[1]))
    smiles.append(content[2].strip())

molecules = []
for s in smiles:
    molecules.append(Chem.MolFromSmiles(s))

__all__ = ['values', 'smiles', 'molecules']