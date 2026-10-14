import gymnasium as gym
import numpy as np

class AstEnv(gym.Env):
    ''' Gymnasium wrapper around a sim interface (SimInterface or MarAstSimInterface), so
        that a DRL solver can search in the same simulators as MCTS. The agent's action in
        [-1,1] is mapped to an action seed in [0,1], and the reward is the sim's own reward.
    '''

    def __init__(self, simInterface, episodeCallback = None) -> None:
        self.simInterface = simInterface
        self.episodeCallback = episodeCallback  # Called with (action trace, G, e) when an episode ends
        self.simInterface.reset_sim()
        observationSize = len(self.simInterface.get_observation())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observationSize,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.G = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.simInterface.reset_sim()
        self.G = 0
        return self.get_observation(), {}

    def step(self, action):
        reward = self.simInterface.step(self.get_seed_from_action(action))
        self.G += reward
        terminal = self.simInterface.is_terminal()
        if terminal and self.episodeCallback is not None:
            self.episodeCallback(list(self.simInterface.get_state()), self.G, self.simInterface.is_failure_episode())
        return self.get_observation(), reward, terminal, False, {}

    def get_observation(self):
        return np.array(self.simInterface.get_observation(), dtype=np.float32)

    def get_seed_from_action(self, action) -> float:
        seed = (float(action[0]) + 1)/2
        return min(max(seed, 0.001), 0.999)  # Never exactly 0 or 1. A full jump between them has p = 0 in SimInterface
