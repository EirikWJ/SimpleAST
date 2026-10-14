import json, math

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from sac.astEnv import AstEnv
from sim.simInterface import SimInterface

REPO_ROOT = Path(__file__).resolve().parents[1]

class SACController:

    def __init__(self, simFactory = None, parameterFile = "parameters_sac.json") -> None:
        if simFactory is None:  # simFactory creates the sim interface to search in, e.g. MarAstSimInterface
            simFactory = SimInterface
        self.simFactory = simFactory
        self.simInterface = simFactory()
        self.endStates = []
        self.crashStates = []
        self.bestState = None
        self.bestReward = -math.inf
        self.Gs = []
        self.policyState = None  # Action trace and reward of the trained policy, run without exploration noise
        self.policyReward = None
        with open(REPO_ROOT / parameterFile) as f:
            self.params = json.load(f)
        params = self.params
        self.env = AstEnv(self.simInterface, self._record_episode)
        self.model = SAC("MlpPolicy", self.env,
                         learning_rate=params["learning_rate"],
                         buffer_size=params["buffer_size"],
                         learning_starts=params["learning_starts"],
                         batch_size=params["batch_size"],
                         gamma=params["gamma"],
                         tau=params["tau"],
                         gradient_steps=params["gradient_steps"],
                         policy_kwargs={"net_arch": params["net_arch"]},
                         seed=params["seed"],
                         device="cpu")

    def loop(self, numberOfLoops):  # Trains the policy for numberOfLoops episodes. Every episode is one simulation
        timeStart = datetime.now()
        self.model.learn(total_timesteps=10**9, callback=StopTrainingOnMaxEpisodes(numberOfLoops))
        runTime = datetime.now() - timeStart
        with open('crashStatesSac.json', 'w') as f:
            json.dump(self.crashStates, f, ensure_ascii=False, indent=4)
        if self.params["model_path"] is not None:
            modelPath = REPO_ROOT / self.params["model_path"]
            modelPath.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(modelPath)
        self.policyState, self.policyReward = self.run_policy()
        self.stats(self.Gs, runTime)
        return self.bestState, self.bestReward

    def run_policy(self):  # One episode with the trained policy's most likely action in every step
        self.simInterface.reset_sim()
        G = 0
        terminal = False
        while not terminal:
            action, _ = self.model.predict(self.env.get_observation(), deterministic=True)
            G += self.simInterface.step(self.env.get_seed_from_action(action))
            terminal = self.simInterface.is_terminal()
        return list(self.simInterface.get_state()), G

    def _record_episode(self, state, G, e) -> None:
        i = len(self.Gs)
        if (i%1000 == 0):
            print(i)
        self.endStates.append(state)
        if e:
            self.crashStates.append(tuple(state))
        if G > self.bestReward:
            self.bestState = state
            self.bestReward = G
            print(f'Score {round(G, 2)} found at iteration {i}')
        self.Gs.append(G)

    def stats(self, Gs, runTime):
        print("-"*35)
        print(f'Runtime         | {runTime}')
        print(f'{"Episodes":<25} | {len(Gs)}')
        print(f'{"Best reward found":<25} | {round(self.bestReward, 2):4}')
        print(f'{"Nr of crash states found":<25} | {len(self.crashStates):}')
        print(f'{"Unique crash states found":<25} | {len(set(self.crashStates))}')
        print(f'{"Best action trace":<25} | {self.bestState}')
        policyCrashed = self.simInterface.is_failure_episode()  # The sim still holds the episode from run_policy
        print(f'{"Trained policy reward":<25} | {round(self.policyReward, 2):4} ({"collision" if policyCrashed else "no collision"})')
        print(f'{"Trained policy trace":<25} | {self.policyState}')
        printSim = self.simFactory()
        printSim.set_state(self.bestState)
        printSim.plot()
        printSim.set_state(self.policyState)
        printSim.plot()
        index = range(len(Gs))
        window = max(1, len(Gs)//20)
        rollingMean = np.convolve(Gs, np.ones(window)/window, mode="valid")
        plt.plot(index, Gs, alpha = 0.4, label = "Total reward of episode")
        plt.plot(index[window-1:], rollingMean, label = f'Mean of last {window} episodes')
        plt.axhline(self.bestReward, color = "gray", linestyle = "--", label = f'Best reward ({round(self.bestReward, 2)})')
        plt.title("Reward over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.legend()
        plt.show()
