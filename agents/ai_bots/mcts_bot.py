"""
MCTS Bot
使用蒙特卡洛树搜索算法
"""

import time
import random
import math
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config
import copy


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.state.get_valid_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.clone()
        next_state.step(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=math.sqrt(2)):
        choices = [
            (child.value / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits), child)
            for child in self.children
        ]
        return max(choices, key=lambda x: x[0])[1]

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)  # 对手视角

    def rollout(self):
        current_state = self.state.clone()
        while not current_state.is_terminal():
            actions = current_state.get_valid_actions()
            if not actions:
                break
            action = random.choice(actions)
            current_state.step(action)
        winner = current_state.get_winner()
        if winner == self.state.current_player:
            return 1
        elif winner is None:
            return 0
        else:
            return -1


class MCTSBot(BaseAgent):
    def __init__(self, name="MCTSBot", player_id=1, simulation_count=100):
        super().__init__(name, player_id)
        self.simulation_count = simulation_count
        ai_config = config.AI_CONFIGS.get('mcts', {})
        self.simulation_count = ai_config.get('simulation_count', simulation_count)
        self.timeout = ai_config.get('timeout', 10)

    def get_action(self, observation, env):
        root_state = env.game.clone()
        root_node = MCTSNode(root_state)

        end_time = time.time() + self.timeout
        simulations = 0

        while time.time() < end_time and simulations < self.simulation_count:
            node = root_node

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            result = node.rollout()

            # Backpropagation
            node.backpropagate(result)

            simulations += 1

        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.action
