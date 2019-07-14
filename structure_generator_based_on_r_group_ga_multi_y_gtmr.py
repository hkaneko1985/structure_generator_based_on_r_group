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
from gtm import GTM
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings('ignore')

target_y_value = [1, -60, 30]  # y-target for inverse analysis

dataset = pd.read_csv('molecules_with_multi_y.csv', index_col=0)  # SMILES 付きデータセットの読み込み
target_ranges = pd.read_csv('settings_of_target_ys.csv', index_col=0)  # 各 y の目標範囲の読み込み
file_name_of_main_fragments = 'sample_main_fragments_logS.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 'r_group' 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
deleting_descriptor_names = ['Ipc', 'Kappa3']
#deleting_descriptor_names = []
number_of_iteration_of_ga = 10  # GA による構造生成を何回繰り返すか (number_of_iteration_of_ga × number_of_population) の数だけ化学構造が得られます

shape_of_map = [30, 30]
shape_of_rbf_centers = [8, 8]
variance_of_rbfs = 0.03125
lambda_in_em_algorithm = 0.5
number_of_iterations = 300
display_flag = 1
number_of_population = 30  # GA の個体数
number_of_generation = 50  # GA の世代数
probability_of_crossover = 0.5
probability_of_mutation = 0.2
threshold_of_variable_selection = 0.5
minimum_number = -10 ** 30

smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1:]  # 物性・活性などの目的変数
numbers_of_y = np.arange(y.shape[1])

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

variables = pd.concat([y, x], axis=1)
numbers_of_x = np.arange(numbers_of_y[-1] + 1, variables.shape[1])

# standardize x and y
autoscaled_variables = (variables - variables.mean(axis=0)) / variables.std(axis=0, ddof=1)
autoscaled_target_y_value = (target_y_value - variables.mean(axis=0)[numbers_of_y]) / variables.std(axis=0, ddof=1)[
    numbers_of_y]

# construct GTMR model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag)
model.fit(autoscaled_variables)

if model.success_flag:
    # calculate of responsibilities
    responsibilities = model.responsibility(autoscaled_variables)
    means = responsibilities.dot(model.map_grids)
    modes = model.map_grids[responsibilities.argmax(axis=1), :]

    mean_of_estimated_mean_of_y, mode_of_estimated_mean_of_y, responsibilities_y, py = \
        model.gtmr_predict(autoscaled_variables.iloc[:, numbers_of_x], numbers_of_x, numbers_of_y)

    plt.rcParams['font.size'] = 18
    for index, y_number in enumerate(numbers_of_y):
        predicted_y_test = mode_of_estimated_mean_of_y[:, index] * variables.iloc[:, y_number].std() + variables.iloc[:,
                                                                                                       y_number].mean()
        # yy-plot
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(variables.iloc[:, y_number], predicted_y_test)
        y_max = np.max(np.array([np.array(variables.iloc[:, y_number]), predicted_y_test]))
        y_min = np.min(np.array([np.array(variables.iloc[:, y_number]), predicted_y_test]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Estimated Y')
        plt.show()
        # r2, RMSE, MAE
        print(
            'r2: {0}'.format(float(1 - sum((variables.iloc[:, y_number] - predicted_y_test) ** 2) / sum(
                (variables.iloc[:, y_number] - variables.iloc[:, y_number].mean()) ** 2))))
        print('RMSE: {0}'.format(float((sum((variables.iloc[:, y_number] - predicted_y_test) ** 2) / len(
            variables.iloc[:, y_number])) ** 0.5)))
        print('MAE: {0}'.format(float(sum(abs(variables.iloc[:, y_number] - predicted_y_test)) / len(
            variables.iloc[:, y_number]))))

        # plot the mean of responsibilities
        plt.scatter(means[:, 0], means[:, 1], c=variables.iloc[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mean)')
        plt.ylabel('z2 (mean)')
        plt.show()
        # plot the mode of responsibilities
        plt.scatter(modes[:, 0], modes[:, 1], c=variables.iloc[:, y_number])
        plt.colorbar()
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('z1 (mode)')
        plt.ylabel('z2 (mode)')
        plt.show()

    # GTMR prediction for inverse analysis
    autoscaled_mean_of_estimated_mean_of_x, autoscaled_mode_of_estimated_mean_of_x, responsibilities_y, py = \
        model.gtmr_predict(autoscaled_target_y_value, numbers_of_y, numbers_of_x)

    # Check results of inverse analysis
    print('Results of inverse analysis')
    mean_of_estimated_mean_of_x = pd.DataFrame(autoscaled_mean_of_estimated_mean_of_x, columns=x.columns) * x.std(
        axis=0, ddof=1) + x.mean(axis=0)
    mode_of_estimated_mean_of_x = pd.DataFrame(autoscaled_mode_of_estimated_mean_of_x, columns=x.columns) * x.std(
        axis=0, ddof=1) + x.mean(axis=0)
    print('estimated x-mode: {0}'.format(mode_of_estimated_mean_of_x))

    estimated_x_mean_on_map = responsibilities_y.dot(model.map_grids)
    estimated_x_mode_on_map = model.map_grids[np.argmax(responsibilities_y), :]
    print('estimated x-mode on map: {0}'.format(estimated_x_mode_on_map))

    plt.scatter(modes[:, 0], modes[:, 1], c='blue')
    plt.scatter(estimated_x_mode_on_map[0], estimated_x_mode_on_map[1], c='red', marker='x', s=100)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('z1 (mode)')
    plt.ylabel('z2 (mode)')
    plt.show()

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
            distance = (((autoscaled_mode_of_estimated_mean_of_x[0, :] - autoscaled_x_prediction.values[0,
                                                                         :]) ** 2).sum()) ** 0.5
            value = 1 / distance
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
