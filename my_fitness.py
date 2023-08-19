# -*- encoding: utf-8 -*-
import numpy as np
import random
from pymoo.core.survival import Survival


class MyFitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        # print('-------doing survival-------')
        F = pop.get("F")
        # F = np.stack((F[:,0],F[:,1:3].sum(axis=1)+F[:,-7:].sum(axis=1),F[:,3:5].sum(axis=1)), axis=1)
        S = np.lexsort(F.transpose())[::-1]
        pop.set("rank", np.argsort(S))
        # front = S[:int(n_survive*0.9)]
        # back = S[int(n_survive*0.9)+1:]
        # random.shuffle(back)
        # new_group = np.concatenate((front, back))
        return pop[S[:n_survive]]
    # return pop[S[:n_survive]]