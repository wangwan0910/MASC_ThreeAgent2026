# -*- coding: utf-8 -*-
"""Factory: most upstream stage, supplied by an unlimited external supplier."""

from .base_agent import BaseAgent


class Factory(BaseAgent):
    name = "factory"
    holding_cost = 1000
    processing_time = 1
    # service time to the warehouse ST0 (the policy's choice is overridden): 0 if the factory has the warehouse's
    # order in stock, otherwise 1 (its own replenishment from the external supplier): warehouse lead time 3 + ST0
    stock_based_service = True
    # the service time is set by the environment, so the policy has no service time choice (1 value)
    action_nvec = (1, 31, 1)
