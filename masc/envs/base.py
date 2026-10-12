# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 09:33:41 2025

@author: wangw

Serial supply chain environment (single planner), shared by the stochastic demand environments
(masc.envs.stochastic) and the batch demand environments (masc.envs.batch), which differ only in gen_demand().
The agents (factory, warehouse, retailer) are defined in masc/agents/, one file per agent, all derived from the
template masc/agents/base_agent.py. To add an agent, create a new agent there and add it to make_agents().
The customer demand is part of the environment (masc/demand.py), not an agent.
"""

from gymnasium.spaces import MultiDiscrete
import datetime
import os
import gymnasium
import csv
import numpy as np
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import logging

from ..agents import BaseAgent, Factory, Warehouse, Retailer



logger = logging.getLogger(__name__)


ep_count = 0


# names used in the csv columns for the state / action fields of BaseAgent
STATE_CSV_NAMES = {
    "inventory": "inventory",
    "reorder_point": "reorderpoint",
    "pending_delivery": "pendingDelivery",
    "days_to_delivery": "daysDelivery",
    "stockout_cum": "stockout_cum",          # stockouts accumulated since the last action (reset in execute)
    "processing_time": "processingTimes",
    "stockout_daily": "stockout_daily",      # stockouts of this day
    "capacity": "capacity",
    "stockout_total": "stockout_total",      # stockouts of the episode
}
ACTION_CSV_NAMES = {
    "service_time": "serviceTime",
    "order_to_supplier": "orderToSupplier",
    "reorder_point": "reorderPoint",
}


# POSingAgent1W1F is the environment class for the serial supply chain inventory management problem,
# by default with one factory, one warehouse, and one retailer, controlled by a single planner (the RL policy).
class POSingAgent1W1F:
    # mean and standard deviation of the customer demand (normal distribution), see masc/demand.py
    muDemand = 2
    stdDemand = 1
# The state is a (len(STATE_FIELDS), n_stage) matrix, one column per agent; the action is a (len(ACTION_FIELDS), n_stage) matrix.
    def __init__(self, *args, **kwargs):

        # demand parameters: class attributes muDemand / stdDemand, set by the subclasses
        self.T = 100
        # episode terminates after this many days with a retailer stockout
        self.max_stockout_days = 12
        self.re_num = 0

        # agents from the most upstream to the most downstream stage
        self.agents = self.make_agents()
        assert isinstance(self.agents[-1], Retailer), "the most downstream agent must serve the customer demand"
        self.retailer = self.agents[-1]
        self.n_stage = len(self.agents)

        # spaces are flattened row by row: [field0 of all agents, field1 of all agents, ...]
        self.action_space = MultiDiscrete(
            np.array([a.action_nvec for a in self.agents]).T.flatten(), dtype = np.int32)
        self.observation_space = MultiDiscrete(
            np.array([a.observation_nvec() for a in self.agents]).T.flatten(), dtype = np.int32)

        self.r = 0
        self.info =  {}
        self.total_steps = 0
        # state / histories are initialised in reset()
        self.reset()

    # the supply chain: override to add / remove / replace agents
    def make_agents(self):
        return [Factory(), Warehouse(), Retailer()]

    # customer demand of one day: defined by the subclasses (stochastic: masc.envs.stochastic, batch: masc.envs.batch)
    def gen_demand(self):
        raise NotImplementedError

    # supplier of each agent: the previous agent, None (unlimited external supplier) for the first one
    def supplier(self, k):
        return self.agents[k - 1] if k > 0 else None

    # current state matrix, one column per agent
    @property
    def s(self):
        return np.stack([a.observation() for a in self.agents], axis=1)


        # natural state update
# pre_step function to update the state based on the demand, and return the new state and action trigger for taking action.
    def Pre_step(self, demand):
        for a in self.agents:
            # daily stockout record starts from 0 every day
            a.start_day()

        # deliveries, from upstream to downstream: an agent can only ship what it has in stock
        for k, a in enumerate(self.agents):
            a.receive_delivery(self.supplier(k))

        # customer demand at the retailer
        if self.retailer.serve_demand(demand):
            self.stockoutCount += 1

        # trigger for taking action: retailer's inventory <= reorderPoint & not waiting for any delivery
        actionTrigger = self.retailer.needs_order()
        self.actionTrigger = actionTrigger

        self.day+= 1

        # per-stage capacity limit
        for a in self.agents:
            a.clip_inventory()
        return (self.s, actionTrigger)

 # cost of each agent on a reorder day: its inventory holding cost + its stockouts since the last action (state[4])
    def agent_costs(self):
        return [a.cost(a.stockout_cum) for a in self.agents]

 # reward function: negative total cost of all agents (holding cost + stockout cost).
    def reward(self):
        self.r = -sum(self.agent_costs())
        return self.r

    # action matrix that is really applied: fixed service times override the policy's choice
    def effective_action(self, action):
        action = action.copy()
        for k, a in enumerate(self.agents):
            action[0, k] = a.service_time(action[:, k])
        return action

# execute function to update the state based on the action taken by the planner agent, and return the new state.
    def execute(self, action):
        # lead time of agent k = service time of its supplier + its own processing time
        supplier_service_time = 0  # the external supplier of the first agent has 0 service time
        for k, a in enumerate(self.agents):
            a.place_order(action[:, k], supplier_service_time)
            supplier_service_time = a.service_time(action[:, k])
        # reset actionTrigger
        self.actionTrigger = False

        return self.s



# seed function to set random seed for reproducibility.
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



# step function to take action and return new state, reward, terminated, truncated and info.
    # A day is split in two halves so that the multi-agent wrapper (marl_env.py) can let the agents act only on
    # reorder days: begin_day() (demand, deliveries, trigger) and end_day(action) (reward, execute action).
    def step(self, a):
        self.begin_day()
        return self.end_day(a)

    # first half of a day: customer demand and deliveries; returns True if the retailer triggers an order
    def begin_day(self):

        global ep_count

        ep_count += 1
        self.t += 1

        # customer demand (part of the environment, not an agent)
        self.demand = self.gen_demand()

        # state transition
        # actionTrigger if retailer hits reorder point
        self.state1, self.actionTrigger = self.Pre_step(self.demand)
        return self.actionTrigger

    # second half of a day: reward, and the action is executed if the retailer triggered an order
    def end_day(self, a):
        # copy so that the caller's action array is not modified below
        action = np.array(a).reshape(len(BaseAgent.ACTION_FIELDS), self.n_stage).copy()
        state1, actionTrigger, demand = self.state1, self.actionTrigger, self.demand
        # sparse reward: the cost is only given on reorder (action trigger) days
        reward = self.reward()
        """
        execute action
        """

        # check if action can be taken
        if actionTrigger:
            # planner decides action
            action = self.effective_action(action)
            # execute action
            state = self.execute(action)
        else:
            action = np.zeros_like(action)
            state = state1
            reward = 0

        # keep track of reward
        self.reward_history = np.append(self.reward_history , [reward], axis=0)

        self.total_steps += 1

        self.state_history = np.append(self.state_history, [state], axis=0)
        self.demand_history = np.append(self.demand_history , [demand], axis=0)
        self.action_history = np.append(self.action_history , [action], axis=0)
        self.actionTriggerHistory = np.append(self.actionTriggerHistory, [actionTrigger], axis=0)


        terminated, truncated = self.is_terminated(), self.is_truncated()

        #self.render()
        #filename = f"./STO{self.muDemand}T{self.stdDemand}.csv"
        #self.save_to_csv(filename)

        return state.flatten(), self.reward_history[-1], terminated, truncated, {}

# check if episode is done
    def is_done(self):
        return self.is_terminated() or self.is_truncated()

    # terminated: failure, too many stockouts
    def is_terminated(self):
        return self.stockoutCount >= self.max_stockout_days

    # truncated: time limit T reached without failing
    def is_truncated(self):
        return (self.day >= self.T) and not self.is_terminated()

# reset the environment to initial state, and return the initial observation.
    def reset(self, seed=None, options=None):
        if seed is not None:
            # DemandGenerator samples from np.random
            np.random.seed(seed)
            self._seed(seed)
        self.re_num += 1
        self.t = 0
        self.stockoutCount = 0
        self.day = 0
        self.r = 0
        self.total_steps = 0
        self.actionTrigger = False

        for a in self.agents:
            a.reset()
        s = self.s

        # clear histories of the previous episode
        self.state_history = [s.copy()]
        initial_action = np.zeros((len(BaseAgent.ACTION_FIELDS), self.n_stage), dtype=int)
        initial_action[0] = [a.fixed_service_time or 0 for a in self.agents]
        self.action_history = [initial_action]
        self.reward_history = []
        self.demand_history = []
        self.actionTriggerHistory = []

        return s.flatten() ,{}


# save the state history, action history, reward history, and demand history to a csv file for analysis and plotting.
    def save_to_csv(self, filename=None):
        fieldnames = []
        for f in BaseAgent.STATE_FIELDS:
            fieldnames += [f"{a.name}_{STATE_CSV_NAMES[f]}" for a in self.agents]
        for f in BaseAgent.ACTION_FIELDS:
            fieldnames += [f"{a.name}_{ACTION_CSV_NAMES[f]}" for a in self.agents]
        fieldnames += ["customer_demand", "Cost", "retailer_actionTriggerHistory"]

        data = []
        for i in range(len(self.state_history)):
            row = {}
            for r, f in enumerate(BaseAgent.STATE_FIELDS):
                for k, a in enumerate(self.agents):
                    row[f"{a.name}_{STATE_CSV_NAMES[f]}"] = self.state_history[i][r][k]
            for r, f in enumerate(BaseAgent.ACTION_FIELDS):
                for k, a in enumerate(self.agents):
                    row[f"{a.name}_{ACTION_CSV_NAMES[f]}"] = self.action_history[i][r][k]
            # row i is the state after day i; row 0 is the initial state, before any day (no demand / cost / trigger)
            # (index i-1 would be -1 for row 0, i.e. the values of the last day)
            row["customer_demand"] = self.demand_history[i-1] if i > 0 else ""
            row["Cost"] = self.reward_history[i-1] if i > 0 else ""
            row["retailer_actionTriggerHistory"] = self.actionTriggerHistory[i-1] if i > 0 else ""
            data.append(row)

        with open(filename, 'w', newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)




 # render function to plot order, inventory, reorder point, stockouts, and reward over time for each agent.
    def render(self, mode="human", log_dir='plots'):
        current_date = datetime.datetime.now()
        global ep_count
        p = self.T
        p = 100 if p <= 100 else p


        if ep_count % p == 0 and len(self.reward_history) % p == 0:
            start = max(0, len(self.reward_history) - p)
            end = len(self.reward_history)

            fig = plt.figure(figsize=(10, 10))  # Adjusted figure size
            agent_labels = [a.name.capitalize() for a in self.agents]
            rc = self.n_stage - 1  # retailer column


            for i in range(self.n_stage):
                ax = fig.add_subplot(5,self.n_stage,i+1)
                ax.plot(range(start, end), [a[1][i] for a in self.action_history[-100:]], label=f'{agent_labels[i]}', color='gray', linestyle='--', linewidth=0.8)
                ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Order')

            for i in range(self.n_stage):
                ax = fig.add_subplot(5,self.n_stage,i+1+self.n_stage)
                ax.plot(range(start, end), [s[0][i] for s in self.state_history[-100:]], label=f'{agent_labels[i]}', color='b', alpha=0.85, linewidth=0.8)
                ax.legend(loc='upper left', fontsize='x-small', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Inventory')
                ax.set_ylim([-1,33])

            ax = fig.add_subplot(5, 1, 3)
            ax.plot(range(start, end), [s[1][rc] for s in self.state_history[-100:]], label=f'Retailer Reorderpoint', color='b', alpha=0.85, linewidth=0.8)
            ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)


            ax = fig.add_subplot(5, 1, 4)

            ax.plot(range(start, end), [s[6][rc] for s in self.state_history[-100:]], label=f'Retailer Stockout', color='b', alpha=0.85, linewidth=0.8)

            ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)

            ax.set_xlabel('Step')

            ax = fig.add_subplot(5, 1, 5)

            ax.plot(range(start, end), [0 if np.isnan(r) else r for r in self.reward_history[-100:] ], label=f'Profit', color='b', alpha=0.85, linewidth=0.8)

            ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)


            # Save the plot
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                filename = os.path.join(log_dir, f'Step{ep_count}_ep{self.re_num}_V{self.muDemand}T{self.stdDemand}_{current_date.strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
