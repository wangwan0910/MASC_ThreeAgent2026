# -*- coding: utf-8 -*-
"""Customer demand of the supply chain environment (part of the environment, not an agent)."""

import numpy as np


# DemandGenerator generates demand based on a normal distribution with mean muDemand and standard deviation stdDemand, and returns the demand as an integer.
class DemandGenerator():
    def __init__(self, muDemand, stdDemand):

        self.muDemand = muDemand
        self.stdDemand = stdDemand

    def genDemand(self):
        demand = max(0, np.random.normal(self.muDemand, self.stdDemand))
        # make sure demand is integer
        demand = np.floor(demand)

        return demand
