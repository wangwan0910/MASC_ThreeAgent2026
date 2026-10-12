# -*- coding: utf-8 -*-
"""
BaseAgent: template for one stage (agent) of the serial supply chain.

The environment (masc.envs.base.POSingAgent1W1F) holds an ordered list of agents, from the most
upstream stage to the most downstream stage, e.g. [Factory, Warehouse, Retailer].
The supplier of agent k is agent k-1; agent 0 is supplied by an unlimited external supplier.

Each agent is one column of the environment state / action matrices:
    state rows  : STATE_FIELDS   (9 rows)
    action rows : ACTION_FIELDS  (3 rows)

How to add a new agent
----------------------
1. Create a new file in masc/agents/, e.g. distributor.py:

       from .base_agent import BaseAgent

       class Distributor(BaseAgent):
           name = "distributor"
           holding_cost = 50
           processing_time = 2
           fixed_service_time = 2          # service time it gives to its downstream stage
           action_nvec = (3, 20, 1)        # serviceTime, orderToSupplier, reorderPoint

2. Export it in masc/agents/__init__.py.
3. Put it into the chain in the environment's make_agents(), e.g.
       return [Factory(), Warehouse(), Distributor(), Retailer()]
   Observation space, action space, reward, CSV columns and plots follow the agent list automatically.
4. Override a method (e.g. cost(), needs_order()) only if the new agent behaves differently.
"""

import numpy as np


class BaseAgent:
    # ----- parameters: override in subclasses -----
    name = "base"
    capacity = 30                # inventory is clipped to [0, capacity - 1]
    holding_cost = 0             # cost per unit of inventory per period
    stockout_cost = 10000        # cost per unit the agent cannot deliver downstream
    processing_time = 0          # added to the supplier's service time to get the delivery lead time
    init_inventory = 0
    init_reorder_point = 0
    # service time this agent gives to its downstream stage; None = taken from the action
    fixed_service_time = None
    # number of discrete action values for (serviceTime, orderToSupplier, reorderPoint)
    action_nvec = (1, 20, 1)
    max_days_to_delivery = 9
    # observation cap of the episode total stockouts
    max_total_stockout = 200

    # rows of the environment state matrix (one column per agent)
    STATE_FIELDS = (
        "inventory",
        "reorder_point",
        "pending_delivery",
        "days_to_delivery",
        "stockout_cum",       # stockouts since the last order (reset when an order is placed)
        "processing_time",
        "stockout_daily",     # stockouts of the current day
        "capacity",
        "stockout_total",     # stockouts of the whole episode (capped at max_total_stockout)
    )
    # rows of the environment action matrix (one column per agent)
    ACTION_FIELDS = ("service_time", "order_to_supplier", "reorder_point")

    def __init__(self):
        self.reset()

    def reset(self):
        self.inventory = self.init_inventory
        self.reorder_point = self.init_reorder_point
        self.pending_delivery = 0
        self.days_to_delivery = 0
        self.stockout_cum = 0
        self.stockout_daily = 0
        self.stockout_total = 0

    # ----- observation -----
    def observation(self):
        """Column of the state matrix, in STATE_FIELDS order."""
        return np.array([
            self.inventory,
            self.reorder_point,
            self.pending_delivery,
            self.days_to_delivery,
            self.stockout_cum,
            self.processing_time,
            self.stockout_daily,
            self.capacity,
            min(self.stockout_total, self.max_total_stockout),
        ])

    def observation_nvec(self):
        """Number of discrete values of each observation field, in STATE_FIELDS order."""
        n = self.capacity + 1
        # reorder point is only observable for agents that choose it
        reorder_point_n = n if self.action_nvec[2] > 1 else 1
        return [n, reorder_point_n, n, self.max_days_to_delivery + 1, n,
                self.processing_time + 1, n, n, self.max_total_stockout + 1]

    # ----- daily dynamics -----
    def start_day(self):
        self.stockout_daily = 0

    def record_stockout(self, quantity):
        self.stockout_cum += quantity
        self.stockout_daily += quantity
        self.stockout_total += quantity

    def ship(self, requested):
        """Ship to the downstream stage what is in stock; the unfilled part is a stockout of this agent
        (no backlog: the unfilled part is not delivered later). Returns the shipped quantity."""
        shipped = min(requested, self.inventory)
        self.record_stockout(requested - shipped)
        self.inventory -= shipped
        return shipped

    def receive_delivery(self, supplier):
        """Count down the lead time and receive the pending order when it arrives.
        supplier: upstream agent, or None for the unlimited external supplier."""
        if self.days_to_delivery > 0:
            self.days_to_delivery -= 1
        if self.days_to_delivery == 0:
            quantity = self.pending_delivery
            if supplier is not None:
                quantity = supplier.ship(quantity)
            self.inventory += quantity
            self.pending_delivery = 0

    def clip_inventory(self):
        self.inventory = int(np.clip(self.inventory, 0, self.capacity - 1))

    # ----- action -----
    def service_time(self, action):
        """Service time given to the downstream stage. action: this agent's action column."""
        if self.fixed_service_time is not None:
            return self.fixed_service_time
        return int(action[0])

    def place_order(self, action, supplier_service_time):
        """Apply this agent's action column when the planner orders."""
        self.pending_delivery = int(action[1])
        self.days_to_delivery = supplier_service_time + self.processing_time
        self.reorder_point = int(action[2])
        self.stockout_cum = 0

    # ----- reward -----
    def cost(self, stockouts):
        """Holding cost of the current inventory + stockout cost of the given stockouts."""
        return self.inventory * self.holding_cost + stockouts * self.stockout_cost
