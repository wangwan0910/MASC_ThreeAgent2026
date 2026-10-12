# -*- coding: utf-8 -*-
"""Batch demand environments: a demand value ~ N(2, 0.1^2) (rounded down, >= 0) is drawn and repeated batch_size
days (masc.demand.DemandAgent). Same supply chain as the stochastic environments, only the demand differs.
The demand generator is kept across days and episodes.

    Batch_L10 : batch size 10   (BatSupplyEnv-v1, Bat10)
    Batch_L7  : batch size 7    (BatSupplyEnv-v2, Bat7)
    Batch_L3  : batch size 3    (BatSupplyEnv-v3, Bat3)
"""

import gymnasium

from ..demand import DemandAgent
from .base import POSingAgent1W1F


class BatchSupplyChain(POSingAgent1W1F):
    stdDemand = 0.1

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        # batch demand generator, kept across days and episodes
        self.customer = DemandAgent(self.muDemand, self.stdDemand, self.batch_size)
        super().__init__(*args, **kwargs)

    # customer demand of one day: batch demand (the stochastic environments differ only here)
    def gen_demand(self):
        return self.customer.genDemand()


class POSAgent1W1F_V1T1(BatchSupplyChain, gymnasium.Env):
    def __init__(self, batch_size=10, *args, **kwargs):
        super().__init__(batch_size, *args, **kwargs)


# batch_size is set to 10.
class Batch_L10(BatchSupplyChain, gymnasium.Env):
    def __init__(self, batch_size=10, *args, **kwargs):
        super().__init__(batch_size, *args, **kwargs)


# batch_size is set to 7.
class Batch_L7(BatchSupplyChain, gymnasium.Env):
    def __init__(self, batch_size=7, *args, **kwargs):
        super().__init__(batch_size, *args, **kwargs)


# batch_size is set to 3.
class Batch_L3(BatchSupplyChain, gymnasium.Env):
    def __init__(self, batch_size=3, *args, **kwargs):
        super().__init__(batch_size, *args, **kwargs)
