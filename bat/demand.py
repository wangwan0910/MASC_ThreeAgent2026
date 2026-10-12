# -*- coding: utf-8 -*-
"""Customer demand of the batch supply chain environment (part of the environment, not an agent).
The stochastic environment (sto directory) draws a new demand every day; here a demand value is drawn
from the same normal distribution and repeated batch_size days."""

import numpy as np


# demand agent that generates demand for each episode, which is generated from a normal distribution with mean muDemand and standard deviation stdDemand. The demand is generated in batches based on the normal distribution where each batch contains batch_size identical demand values. The random seed can be set for reproducibility.
class DemandAgent():
    def __init__(self,muDemand,stdDemand,batch_size,random_seed=None):
        self.index = 0
        self.batch_size = batch_size
        self.muDemand = muDemand
        self.stdDemand = stdDemand
        self.random_seed = random_seed
        self.generator = self.generate_data()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    # generate demand data in batches based on the normal distribution where each batch contains batch_size identical demand values.
    def generate_data(self):
        while True:
            value = np.maximum(0,np.floor(np.random.normal(self.muDemand,self.stdDemand)))
            values = np.repeat(value,self.batch_size)
            for value in values:
                yield value
    # get demand value from the generator.
    def _getdemand(self):
        return next(self.generator)
    # generate demand value for each episode.
    def genDemand(self):

        return self._getdemand()
