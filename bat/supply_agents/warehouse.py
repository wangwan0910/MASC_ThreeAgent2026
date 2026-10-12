# -*- coding: utf-8 -*-
"""Warehouse: middle stage, supplied by the factory."""

from .base_agent import BaseAgent


class Warehouse(BaseAgent):
    name = "warehouse"
    holding_cost = 5
    processing_time = 3
    # service time to the retailer (the policy's choice is overridden)
    fixed_service_time = 4
    action_nvec = (5, 20, 1)
