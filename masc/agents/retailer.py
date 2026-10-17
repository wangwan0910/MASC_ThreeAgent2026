# -*- coding: utf-8 -*-
"""Retailer: most downstream stage, serves the customer demand and triggers the orders."""

from .base_agent import BaseAgent


class Retailer(BaseAgent):
    name = "retailer"
    holding_cost = 1000
    # transport from the warehouse: 1 day when the warehouse ships from stock
    processing_time = 1
    init_inventory = 10
    init_reorder_point = 5
    fixed_service_time = 0
    # the retailer chooses its reorder point (0-29); order quantity 0-30
    action_nvec = (1, 31, 30)

    def lead_time(self, supplier_service_time):
        """Warehouse in stock: processing time (1 day). Out of stock: the goods are forwarded on the day the
        warehouse's replenishment arrives (3 days if the factory had them in stock, 4 days otherwise)."""
        return max(self.processing_time, supplier_service_time)

    def serve_demand(self, demand):
        """Serve the customer demand; returns True if there was a stockout."""
        shipped = self.ship(int(demand))
        return shipped < demand

    def needs_order(self):
        """Order when the inventory hits the reorder point and no delivery is pending."""
        return (self.inventory <= self.reorder_point) and (self.days_to_delivery == 0)
