# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import random
import warnings

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structure_generator
from deap import base
from deap import creator
from deap import tools
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel

warnings.filterwarnings('ignore')

dataset = pd.read_csv('molecules_with_multi_y.csv', index_col=0)  # SMILES 付きデータセットの読み込み
target_ranges = pd.read_csv('settings_of_target_ys.csv', index_col=0)  # 各 y の目標範囲の読み込み
file_name_of_main_fragments = 'sample_main_fragments_logS.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 'r_group' 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
deleting_descriptor_names = ['Ipc', 'Kappa3']
#deleting_descriptor_names = []
number_of_iteration_of_ga = 10  # GA による構造生成を何回繰り返すか (number_of_iteration_of_ga × number_of_population) の数だけ化学構造が得られます

number_of_population = 30  # GA の個体数
number_of_generation = 50  # GA の世代数
probability_of_crossover = 0.5
probability_of_mutation = 0.2
threshold_of_variable_selection = 0.5
minimum_number = -10 ** 30

smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1:]  # 物性・活性などの目的変数
numbers_of_y = y.shape[1]

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = []  # ここに計算された記述子の値を追加
print('分子の数 :', len(smiles))
for index, smiles_i in enumerate(smiles):
    print(index + 1, '/', len(smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
original_x = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)
if deleting_descriptor_names is not None:
    original_x = original_x.drop(deleting_descriptor_names, axis=1)
original_x = original_x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = original_x.isnull().any()  # NaN を含む変数
original_x = original_x.drop(original_x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = original_x.std() == 0
x = original_x.drop(original_x.columns[std_0_variable_flags], axis=1)

# オートスケーリング
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()

# modeling
models = []
for y_number in range(numbers_of_y):
    model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), alpha=0)
    model.fit(autoscaled_x, autoscaled_y.iloc[:, y_number])  # モデルの構築
    models.append(model)

# 構造生成
main_molecules = [molecule for molecule in Chem.SmilesMolSupplier(file_name_of_main_fragments,
                                                                  delimiter='\t', titleLine=False)
                  if molecule is not None]
fragment_molecules = [molecule for molecule in Chem.SmilesMolSupplier(file_name_of_sub_fragments,
                                                                      delimiter='\t', titleLine=False)
                      if molecule is not None]

creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(len(fragment_molecules) + 1)
max_boundary = np.ones(len(fragment_molecules) + 1) * 1.0


def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary):
        index.append(random.uniform(min, max))
    return index


toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    individual_array = np.array(individual)
    generated_smiles = structure_generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                                individual_array)
    generated_molecule = Chem.MolFromSmiles(generated_smiles)
    if generated_molecule is not None:
        AllChem.Compute2DCoords(generated_molecule)
        descriptors_of_generated_molecule = descriptor_calculator.CalcDescriptors(generated_molecule)
        descriptors_of_generated_molecule = pd.DataFrame(descriptors_of_generated_molecule, index=descriptor_names)
        descriptors_of_generated_molecule = descriptors_of_generated_molecule.T
        if deleting_descriptor_names is not None:
            descriptors_of_generated_molecule = descriptors_of_generated_molecule.drop(deleting_descriptor_names,
                                                                                       axis=1)
        descriptors_of_generated_molecule = descriptors_of_generated_molecule.drop(
            descriptors_of_generated_molecule.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
        descriptors_of_generated_molecule = descriptors_of_generated_molecule.drop(
            descriptors_of_generated_molecule.columns[std_0_variable_flags], axis=1)
        descriptors_of_generated_molecule = descriptors_of_generated_molecule.replace(np.inf, np.nan).fillna(
            np.nan)  # inf を NaN に置き換え
        if descriptors_of_generated_molecule.isnull().sum(axis=1)[0] > 0:
            value = minimum_number
        else:
            # オートスケーリング
            autoscaled_x_prediction = (descriptors_of_generated_molecule - x.mean()) / x.std()
            value = 0
            for y_number in range(numbers_of_y):
                estimated_y, std_of_estimated_y = models[y_number].predict(autoscaled_x_prediction, return_std=True)
                estimated_y = estimated_y[0] * y.std()[y_number] + y.mean()[y_number]
                std_of_estimated_y = std_of_estimated_y[0] * y.std()[y_number]
                if std_of_estimated_y <= 0:
                    value += minimum_number
                else:
                    probability_of_y = norm.cdf(target_ranges.iloc[1, y_number], loc=estimated_y,
                                                scale=std_of_estimated_y) - norm.cdf(
                        target_ranges.iloc[0, y_number], loc=estimated_y, scale=std_of_estimated_y)
                    #                    print(estimated_y, std_of_estimated_y, probability_of_y)
                    if probability_of_y <= 0:
                        value += minimum_number
                    else:
                        value += np.log(probability_of_y)

    else:
        value = minimum_number

    return value,


toolbox.register('evaluate', evalOneMax)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

generated_smiles_all = []
estimated_y_all = []
for iteration_number in range(number_of_iteration_of_ga):
    print(iteration_number + 1, '/', number_of_iteration_of_ga)
    # random.seed(100)
    random.seed()
    pop = toolbox.population(n=number_of_population)

    print('Start of evolution')

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print('  Evaluated %i individuals' % len(pop))

    for generation in range(number_of_generation):
        print('-- Generation {0} --'.format(generation + 1))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('  Evaluated %i individuals' % len(invalid_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print('  Min %s' % min(fits))
        print('  Max %s' % max(fits))
        print('  Avg %s' % mean)
        print('  Std %s' % std)

    print('-- End of (successful) evolution --')

    for each_pop in pop:
        if each_pop.fitness.values[0] > minimum_number / 2:
            estimated_y_all.append(each_pop.fitness.values[0])
            each_pop_array = np.array(each_pop)
            smiles = structure_generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                              each_pop_array)
            generated_smiles_all.append(smiles)

estimated_y_all = pd.DataFrame(estimated_y_all, index=generated_smiles_all, columns=['estimated_y'])
estimated_y_all = estimated_y_all.loc[~estimated_y_all.index.duplicated(keep='first'), :]  # 重複したサンプルの最初だけを残す
estimated_y_all.to_csv('generated_molecules.csv')
