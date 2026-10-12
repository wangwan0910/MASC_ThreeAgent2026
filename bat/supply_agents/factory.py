# -*- coding: utf-8 -*-
"""Factory: most upstream stage, supplied by an unlimited external supplier."""

from .base_agent import BaseAgent


class Factory(BaseAgent):
    name = "factory"
    holding_cost = 1000
    processing_time = 1
    # service time to the warehouse (the policy's choice is overridden)
    fixed_service_time = 1
    action_nvec = (2, 20, 1)
