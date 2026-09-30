import json, math

from typing import List, Tuple

from sim.simpleBoatController import SimpleBoatController

class SimInterface:

    def __init__(self) -> None:  # Initiates simulation at state zero
        self.simWorld = SimpleBoatController()
        with open("parameters.json") as f:
            params = json.load(f)
        action_range = params["action_range"]
        self.collision_reward = params["collision_reward"]
        self.totalRange = action_range[1] - action_range[0]
        self.minAction = action_range[0]
        self.actionTrace : List(float) = []
        self.sim_in_endstate : bool = False
        self.lastActionSeed = None

    def step(self, actionSeed) -> Tuple:  # return reward
        action = self._get_action_from_seed(actionSeed)
        p = self._get_transition_probability(actionSeed)
        e, d = self.simWorld.execute_action(action)
        self.actionTrace.append(actionSeed)
        self.sim_in_endstate = self.simWorld.is_endstate()
        self.lastActionSeed = actionSeed
        return self.reward(p, e, d, self.sim_in_endstate)

    def is_terminal(self) -> bool:
        return self.sim_in_endstate

    def is_failure_episode(self) -> bool:
        return self.simWorld.collision_happened

    def get_state(self) -> List:
        return(self.actionTrace)
    
    def set_state(self, state : Tuple) -> None:
        self.reset_sim()
        state = list(state)
        for actionSeed in state:
            self.step(actionSeed)

    def reset_sim(self) -> None:
        self.actionTrace = []
        self.lastActionSeed = None
        self.simWorld.reset_sim()

    def reward(self, # Reward function.
        p,  # Transition probability
        e,  # An episode accured (e.g. boats crashed or NMAC)
        d,  # Closest distance between the boats throughout the simulation
        terminal):  # Simulation has terminated
        if terminal:
            if e:
                return self.collision_reward
            else:
                return -d
        else:
            return math.log(p)

    def plot(self) -> None:
        self.simWorld.plot(self._generate_action_trace(self.actionTrace))

    def _get_transition_probability(self, action):
        if self.lastActionSeed is None:
            return 1
        else:
            return 1-abs(action - self.lastActionSeed)  # TODO: Check if this is correct
    
    def _get_action_from_seed(self, actionSeed):
        return actionSeed * self.totalRange + self.minAction
        
    def _generate_action_trace(self, actionSeedTrace):
        actionTrace = []
        for actionSeed in actionSeedTrace:
            actionTrace.append(self._get_action_from_seed(actionSeed))
        return actionTrace
