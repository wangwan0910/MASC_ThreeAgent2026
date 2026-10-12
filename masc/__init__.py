# -*- coding: utf-8 -*-
"""MASC: multi-agent reinforcement learning (Ray RLlib PPO) for a serial three-stage supply chain
(factory -> warehouse -> retailer), with stochastic and batch customer demand.

Modules
    masc.agents      supply chain stages (BaseAgent template, Factory, Warehouse, Retailer)
    masc.demand      customer demand (DemandGenerator: stochastic, DemandAgent: batch)
    masc.envs        single-planner environments (Stocha_P0/P01/P1, Batch_L10/L7/L3)
    masc.marl_env    multi-agent wrapper (SupplyChainMARLEnv), environment ids, PPO config (build_ppo_config)
    masc.train       single-environment training (masc-train), masc.train_rppo with LSTM (masc-train-rppo)
    masc.continual   continual learning between environments (masc-continual)
    masc.plotting    figures (masc-plot, masc-plot-continual)
"""

__version__ = "0.1.0"
