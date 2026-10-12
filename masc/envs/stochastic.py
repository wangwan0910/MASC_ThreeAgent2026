# -*- coding: utf-8 -*-
"""Stochastic demand environments: a new demand ~ N(muDemand, stdDemand^2) every day (rounded down, >= 0).

    Stocha_P1  : std 1     (StoSupplyEnv-v3, Sto1)
    Stocha_P01 : std 0.1   (StoSupplyEnv-v2, Sto01)
    Stocha_P0  : std 0, the demand is always 2   (StoSupplyEnv-v1, Sto0)
"""

import gymnasium

from ..demand import DemandGenerator
from .base import POSingAgent1W1F


class StochasticSupplyChain(POSingAgent1W1F):
    # customer demand of one day, a new draw every day
    def gen_demand(self):
        return DemandGenerator(self.muDemand, self.stdDemand).genDemand()


class POSAgent1W1F_V1T1(StochasticSupplyChain, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Stochastic demand environments: mean 2, standard deviation 1 / 0.1 / 0 (std 0: the demand is always 2).
# standard deviation 1
class Stocha_P1(StochasticSupplyChain, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.muDemand = 2
        self.stdDemand = 1


# standard deviation 0.1
class Stocha_P01(StochasticSupplyChain, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.muDemand = 2
        self.stdDemand = 0.1


# no standard deviation: the demand is always 2
class Stocha_P0(StochasticSupplyChain, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.muDemand = 2
        self.stdDemand = 0
