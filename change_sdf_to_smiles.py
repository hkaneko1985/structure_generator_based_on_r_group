# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

from rdkit import Chem

# load molecules
molecules = [molecule for molecule in Chem.SDMolSupplier('fragments.sdf') if molecule is not None]
print('number of molecules :', len(molecules))

# change MOL to SMILES
molecules_smiles = []
for molecule in molecules:
    molecules_smiles.append(Chem.MolToSmiles(molecule))

# save SMILES of molecules
str_ = '\n'.join(molecules_smiles)
with open('fragments.smi', 'wt') as writer:
    writer.write(str_)
writer.close()
