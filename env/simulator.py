"""
env/simulator.py — shared environment. Single source of truth for both the
optimization track and the RL track. Do not fork; propose changes via PR.

API contract (draft, finalize jointly):
    reset(scenario_config) -> observation
    step(action) -> observation, reward, done, info
"""


class DecisionIntelligenceEnv:
    def __init__(self, scenario_config: dict):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
