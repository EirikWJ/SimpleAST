import json, math, random

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from plotOutput import finish_figure
from sim.simInterface import SimInterface

class RandomSearchController:
    ''' Baseline solver: every action seed is uniformly random, with no tree and no learning.
        Shows how often random behaviour alone leads to a collision. MCTS and SAC should beat it.
        Same interface as MCTSController and SACController: loop(n) runs n episodes.
    '''

    def __init__(self, simFactory = None, seed = 0, resultsDir = None, onEpisodeEnd = None) -> None:
        self.onEpisodeEnd = onEpisodeEnd  # Called with (sim, G) after every episode, while the sim still holds it
        if simFactory is None:  # simFactory creates the sim interface to search in, e.g. MarAstSimInterface
            simFactory = SimInterface
        self.simFactory = simFactory
        self.simInterface = simFactory()
        self.rng = random.Random(seed)
        self.resultsDir = resultsDir  # Folder for crashStates.json. None: the working directory
        self.crashStates = []  # Action traces, or (start seed, action trace) when the start varies between episodes
        self.bestState = None
        self.bestStartSeed = None
        self.bestReward = -math.inf
        self.Gs = []

    def loop(self, numberOfLoops):
        timeStart = datetime.now()
        sim = self.simInterface
        for i in range(numberOfLoops):
            if (i%1000 == 0):
                print(i)
            sim.reset_sim()
            G = 0
            terminal = False
            while not terminal:  # SimInterface does not clear its end state on reset, so step before checking
                G += sim.step(self.rng.random())
                terminal = sim.is_terminal()
            if self.onEpisodeEnd is not None:
                self.onEpisodeEnd(sim, G)
            state = list(sim.get_state())
            startSeed = getattr(sim, "startSeed", None)  # Only set when the start varies, e.g. the free adversary
            if sim.is_failure_episode():
                self.crashStates.append(tuple(state) if startSeed is None else (startSeed, tuple(state)))
            if G > self.bestReward:
                self.bestState, self.bestStartSeed, self.bestReward = state, startSeed, G
                print(f'Score {round(G, 2)} found at episode {i}')
            self.Gs.append(G)
        runTime = datetime.now() - timeStart
        crashStatesPath = 'crashStatesRandom.json' if self.resultsDir is None else Path(self.resultsDir) / 'crashStates.json'
        with open(crashStatesPath, 'w') as f:
            json.dump(self.crashStates, f, ensure_ascii=False, indent=4)
        self.stats(runTime)
        return self.bestState, self.bestReward

    def stats(self, runTime):
        print("-"*35)
        print(f'Runtime         | {runTime}')
        print(f'{"Episodes":<25} | {len(self.Gs)}')
        print(f'{"Best reward found":<25} | {round(self.bestReward, 2):4}')
        print(f'{"Nr of crash states found":<25} | {len(self.crashStates)}')
        print(f'{"Unique crash states found":<25} | {len(set(self.crashStates))}')
        if self.bestStartSeed is not None:
            print(f'{"Best start seed":<25} | {self.bestStartSeed}')
        print(f'{"Best action trace":<25} | {self.bestState}')
        printSim = self.simFactory()
        if self.bestStartSeed is None:
            printSim.set_state(self.bestState)
        else:
            printSim.set_state(self.bestState, self.bestStartSeed)
        printSim.plot()
        plt.plot(range(len(self.Gs)), self.Gs, linestyle="", marker=".", alpha=0.5, label="Total reward of episode")
        plt.axhline(self.bestReward, color="gray", linestyle="--", label=f'Best reward ({round(self.bestReward, 2)})')
        plt.title("Reward over episodes (random search)")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.legend()
        finish_figure("reward_over_episodes")
