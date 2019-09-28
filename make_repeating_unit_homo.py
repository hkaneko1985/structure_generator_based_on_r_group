# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
from rdkit import Chem

number_of_repeating_unit = 2
inverse = False

main_molecule = Chem.MolFromMolFile('monomer.mol', sanitize=False)

bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]

for repeat in range(number_of_repeating_unit - 1):
    if repeat == 0:
        # make adjacency matrix and get atoms for main molecule
        main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(main_molecule)
        for bond in main_molecule.GetBonds():
            main_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
            main_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())
        main_atoms = []
        for atom in main_molecule.GetAtoms():
            main_atoms.append(atom.GetSymbol())

        r_index_in_main_molecule_old = [index for index, atom in enumerate(main_atoms) if atom == '*']

        added_atoms = main_atoms.copy()
        added_adjacency_matrix = main_adjacency_matrix.copy()
        r_index_in_added_molecule_old = r_index_in_main_molecule_old.copy()

        r_index = r_index_in_main_molecule_old[-1]
        main_atoms[r_index], main_atoms[-1] = main_atoms[-1], main_atoms[r_index]
        main_adjacency_matrix[:, r_index], main_adjacency_matrix[:, -1] = main_adjacency_matrix[:, -1].copy(), main_adjacency_matrix[:, r_index].copy()
        main_adjacency_matrix[r_index, :], main_adjacency_matrix[-1, :] = main_adjacency_matrix[-1, :].copy(), main_adjacency_matrix[r_index, :].copy()
        r_index_in_main_molecule_new = main_adjacency_matrix.shape[0] - 1
        r_bonded_atom_index_in_main_molecule = np.where(main_adjacency_matrix[r_index_in_main_molecule_new, :] != 0)[0][
            0]
        r_bond_number_in_main_molecule = main_adjacency_matrix[
            r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]

        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 0)
        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 1)

        main_atoms.pop(-1)

        main_size = main_adjacency_matrix.shape[0]

        if inverse:
            r_index = r_index_in_added_molecule_old[-1]
        else:
            r_index = r_index_in_added_molecule_old[-2]

        added_atoms[r_index], added_atoms[-1] = added_atoms[-1], added_atoms[r_index]
        added_adjacency_matrix[:, r_index], added_adjacency_matrix[:, -1] = added_adjacency_matrix[:, -1].copy(), added_adjacency_matrix[:, r_index].copy()
        added_adjacency_matrix[r_index, :], added_adjacency_matrix[-1, :] = added_adjacency_matrix[-1, :].copy(), added_adjacency_matrix[r_index, :].copy()
        r_index_in_added_molecule_new = added_adjacency_matrix.shape[0] - 1
        r_bonded_atom_index_in_added_molecule = \
        np.where(added_adjacency_matrix[r_index_in_added_molecule_new, :] != 0)[0][0]

        r_bond_number_in_added_molecule = added_adjacency_matrix[
            r_index_in_added_molecule_new, r_bonded_atom_index_in_added_molecule]

        added_adjacency_matrix = np.delete(added_adjacency_matrix, r_index_in_added_molecule_new, 0)
        added_adjacency_matrix = np.delete(added_adjacency_matrix, r_index_in_added_molecule_new, 1)

        added_atoms.pop(-1)
    else:
        r_index_in_main_molecule_old = [index for index, atom in enumerate(generated_molecule_atoms) if atom == '*']
        if inverse:
            r_index = r_index_in_main_molecule_old[-2]
        else:
            r_index = r_index_in_main_molecule_old[-1]
        main_atoms = generated_molecule_atoms.copy()
        main_adjacency_matrix = generated_adjacency_matrix.copy()

        main_atoms[r_index], main_atoms[-1] = main_atoms[-1], main_atoms[r_index]
        main_adjacency_matrix[:, r_index], main_adjacency_matrix[:, -1] = main_adjacency_matrix[:, -1].copy(), main_adjacency_matrix[:, r_index].copy()
        main_adjacency_matrix[r_index, :], main_adjacency_matrix[-1, :] = main_adjacency_matrix[-1, :].copy(), main_adjacency_matrix[r_index, :].copy()
        r_index_in_main_molecule_new = main_adjacency_matrix.shape[0] - 1
        r_bonded_atom_index_in_main_molecule = np.where(main_adjacency_matrix[r_index_in_main_molecule_new, :] != 0)[0][
            0]
        r_bond_number_in_main_molecule = main_adjacency_matrix[
            r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]

        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 0)
        main_adjacency_matrix = np.delete(main_adjacency_matrix, r_index_in_main_molecule_new, 1)

        main_atoms.pop(-1)

    generated_molecule_atoms = main_atoms[:]
    generated_adjacency_matrix = main_adjacency_matrix.copy()

    main_size = generated_adjacency_matrix.shape[0]
    generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
        [generated_adjacency_matrix.shape[0], added_adjacency_matrix.shape[0]], dtype='int32')]
    generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
        [added_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]
    generated_adjacency_matrix[main_size:, main_size:] = added_adjacency_matrix

    #    for r_index_in_main_molecule_new in range(len(r_index_in_main_molecule_new)):
    #    for r_number_in_added_molecule in range(len(r_index_in_added_molecule_new)):
    generated_adjacency_matrix[
        r_bonded_atom_index_in_main_molecule, r_bonded_atom_index_in_added_molecule + main_size] = r_bond_number_in_main_molecule
    generated_adjacency_matrix[
        r_bonded_atom_index_in_added_molecule + main_size, r_bonded_atom_index_in_main_molecule] = r_bond_number_in_main_molecule

    # integrate atoms
    generated_molecule_atoms += added_atoms

# generate structures 
generated_molecule = Chem.RWMol()
atom_index = []
for atom_number in range(len(generated_molecule_atoms)):
    atom = Chem.Atom(generated_molecule_atoms[atom_number])
    molecular_index = generated_molecule.AddAtom(atom)
    atom_index.append(molecular_index)
for index_x, row_vector in enumerate(generated_adjacency_matrix):
    for index_y, bond in enumerate(row_vector):
        if index_y <= index_x:
            continue
        if bond == 0:
            continue
        else:
            generated_molecule.AddBond(atom_index[index_x], atom_index[index_y], bond_list[bond])

generated_molecule = generated_molecule.GetMol()
generated_molecule_smi = Chem.MolToSmiles(generated_molecule)

with open('generated_structure.smi', 'w') as writer:
    writer.write(generated_molecule_smi + '\n')
writer.close()
