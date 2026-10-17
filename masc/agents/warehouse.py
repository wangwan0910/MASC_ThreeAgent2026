# -*- coding: utf-8 -*-
"""Warehouse: middle stage, supplied by the factory."""

from .base_agent import BaseAgent


class Warehouse(BaseAgent):
    name = "warehouse"
    holding_cost = 5
    processing_time = 3
    # service time to the retailer (the policy's choice is overridden): 0 if the warehouse has the retailer's order
    # in stock, otherwise the days until its own replenishment arrives (3 or 4)
    stock_based_service = True
    # the service time is set by the environment, so the policy has no service time choice (1 value)
    action_nvec = (1, 31, 1)
