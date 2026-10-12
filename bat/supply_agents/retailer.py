# -*- coding: utf-8 -*-
"""Retailer: most downstream stage, serves the customer demand and triggers the orders."""

from .base_agent import BaseAgent


class Retailer(BaseAgent):
    name = "retailer"
    holding_cost = 1000
    processing_time = 0
    init_inventory = 10
    init_reorder_point = 5
    fixed_service_time = 0
    # the retailer chooses its reorder point
    action_nvec = (1, 20, 10)

    def serve_demand(self, demand):
        """Serve the customer demand; returns True if there was a stockout."""
        shipped = self.ship(int(demand))
        return shipped < demand

    def needs_order(self):
        """Order when the inventory hits the reorder point and no delivery is pending."""
        return (self.inventory <= self.reorder_point) and (self.days_to_delivery == 0)
