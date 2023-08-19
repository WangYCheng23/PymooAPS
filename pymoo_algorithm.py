# -*- encoding: utf-8 -*-
import numpy as np
# from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population


from my_mutation import single_day_Mutation
from my_crossover import single_day_Crossover
from my_fitness import MyFitnessSurvival
from utils import df_encode

from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.core.duplicate import HashDuplicateElimination
from pymoo.operators.selection.tournament import compare, TournamentSelection


def weights_compare(a, a_val, b, b_val):
    if a_val[0]>b_val[0]:
        return a
    elif a_val[0]<b_val[0]:
        return b
    else:
        if a_val[1]>b_val[1]:
            return a
        elif a_val[1]<b_val[1]:
            return b
        else:
            if a_val[2]>b_val[2]:
                return a
            elif a_val[2]<b_val[2]:
                return b
            else:
                if a_val[3]>b_val[3]:
                    return a
                elif a_val[3]<b_val[3]:
                    return b
                else:
                    if a_val[4]>b_val[4]:
                        return a
                    elif a_val[4]<b_val[4]:
                        return b
                    else:
                        return np.random.choice([a, b])

def my_comp(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        S[i] = weights_compare(a, pop[a].F, b, pop[b].F)

    return S[:, None].astype(int)

def algorithm_choose(X, data, case='GA', pop_size=50, new=True):
    """算法选择"""
    data_encode = df_encode(data)
    pop = Population.new("X", X)
    if case == "GA":
        algorithm = GA(
            pop_size=pop_size,
            n_offsprings=pop_size, 
            sampling=pop, 
            selection=TournamentSelection(func_comp=my_comp),
            mutation=single_day_Mutation(data_encode),
            crossover=single_day_Crossover(data_encode), 
            survival=MyFitnessSurvival(), 
            eliminate_duplicates=HashDuplicateElimination()
        )
        
    # print('优化策略:', case)
    return algorithm