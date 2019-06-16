# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import random
import sys

import numpy as np
import pandas as pd
import structure_generator
from deap import base
from deap import creator
from deap import tools
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

dataset = pd.read_csv('molecules_with_logS.csv', index_col=0)  # SMILES 付きデータセットの読み込み
file_name_of_main_fragments = 'sample_main_fragments.smi'  # 'r_group' 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 'r_group' 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
number_of_iteration_of_ga = 10  # GA による構造生成を何回繰り返すか (number_of_iteration_of_ga × number_of_population) の数だけ化学構造が得られます

method_name = 'svr'  # 'pls' or 'svr'

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 30  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
number_of_population = 30  # GA の個体数
number_of_generation = 50  # GA の世代数
probability_of_crossover = 0.5
probability_of_mutation = 0.2
threshold_of_variable_selection = 0.5
minimum_number = -10 ** 10

if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))

smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1]  # 物性・活性などの目的変数

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
original_x = original_x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = original_x.isnull().any()  # NaN を含む変数
original_x = original_x.drop(original_x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = original_x.std() == 0
x = original_x.drop(original_x.columns[std_0_variable_flags], axis=1)

# オートスケーリング
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()

if method_name == 'pls':
    # CV による成分数の最適化
    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x), max_number_of_principal_components) + 1):
        # PLS
        model = PLSRegression(n_components=component)  # PLS モデルの宣言
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x, autoscaled_y,
                                                           cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
        r2_in_cv = metrics.r2_score(y, estimated_y_in_cv)  # r2 を計算
        print(component, r2_in_cv)  # 成分数と r2 を表示
        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
        components.append(component)  # 成分数を追加
    optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]
    print('\nCV で最適化された成分数 :', optimal_component_number)
    # PLS
    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
elif method_name == 'svr':
    # グラム行列の分散を最大化することによる γ の最適化
    variance_of_gram_matrix = list()
    for index, ocsvm_gamma in enumerate(svr_gammas):
        print(index + 1, '/', len(svr_gammas))
        gram_matrix = np.exp(-ocsvm_gamma * cdist(autoscaled_x, autoscaled_x, metric='seuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_svr_gamma = svr_gammas[variance_of_gram_matrix.index(max(variance_of_gram_matrix))]
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言

model.fit(autoscaled_x, autoscaled_y)  # モデルの構築
if method_name == 'pls':
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x.columns,
                                                    columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

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
            value = np.ndarray.flatten(model.predict(autoscaled_x_prediction) * y.std() + y.mean())
            value = value[0]
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

    #    best_individual = tools.selBest(pop, 1)[0]
    #    best_value = best_individual.fitness.values[0]
    for each_pop in pop:
        if each_pop.fitness.values[0] is not minimum_number:
            estimated_y_all.append(each_pop.fitness.values[0])
            each_pop_array = np.array(each_pop)
            smiles = structure_generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                              each_pop_array)
            generated_smiles_all.append(smiles)

estimated_y_all = pd.DataFrame(estimated_y_all, index=generated_smiles_all, columns=['estimated_y'])
estimated_y_all.to_csv('generated_molecules.csv')
