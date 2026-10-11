import json, math

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sim.marast.marastPath import REPO_ROOT
from sim.marast.scenario import build_two_ship_env

from env_wrappers.multiship_env.env import AssetInfo

class MarAstSimInterface:
    ''' Same interface as sim.simInterface.SimInterface, but backed by MAR-AST's Ship in
        Transit simulator. The own ship follows its route with SBMPC collision avoidance.
        The action seed steers the adversary ship: seed in [0,1] -> heading offset on its
        LOS heading, held for one action period.
    '''

    def __init__(self, parameterFile = "parameters_marast.json") -> None:  # Initiates simulation at state zero
        with open(REPO_ROOT / parameterFile) as f:
            self.params = json.load(f)
        params = self.params
        self.env, self.assets, self.map_gdfs = build_two_ship_env(params)
        self.ownShip = self.assets[0].ship_model
        self.adversary = self.assets[1].ship_model
        self.stepsPerAction = max(1, round(params["action_period"]/params["time_step"]))
        self.maxOffset = math.radians(params["max_heading_offset_deg"])
        self.offsetSigma = math.radians(params["heading_offset_sigma_deg"])
        self.transition_model = params["transition_model"]
        self.collision_distance = params["collision_distance"]
        self.collision_reward = params["collision_reward"]
        self.miss_distance_scale = params["miss_distance_scale"]
        self.pass_margin = params["pass_margin"]
        self.max_sim_time = params["max_sim_time"]
        self.reset_sim()

    def step(self, actionSeed) -> float:  # return reward
        p = self._get_transition_probability(actionSeed)
        self.adversary.heading_offset = self._get_action_from_seed(actionSeed)
        for _ in range(self.stepsPerAction):
            self.env.step()
            self._record_positions()
            if self._check_endstate():
                break
        self.actionTrace.append(actionSeed)
        self.lastActionSeed = actionSeed
        return self.reward(p, self.is_failure_episode(), self.closest_distance, self.sim_in_endstate)

    def is_terminal(self) -> bool:
        return self.sim_in_endstate

    def is_failure_episode(self) -> bool:
        return self.collision_happened

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
        self.sim_in_endstate = False
        self.collision_happened = False
        self.terminal_reason = None
        self.env.reset(seed=self.params["env_seed"])
        # MAR-AST leaves some state behind between episodes. Clear it, so that an action
        # trace always replays to the same outcome:
        for asset in self.assets:  # env.reset() reuses the info object of the previous episode
            ship = asset.ship_model
            asset.info = AssetInfo(
                current_north=ship.north, current_east=ship.east, current_yaw_angle=ship.yaw_angle,
                forward_speed=ship.forward_speed, sideways_speed=ship.sideways_speed,
                name_tag=ship.name_tag, ship_length=ship.l_ship, ship_width=ship.w_ship)
            ship.sbmpc._params.P_ca_last_ = 1.0  # SBMPC remembers its last chosen offsets
            ship.sbmpc._params.Chi_ca_last_ = 0.0
            ship.sbmpc.active = False
        self.adversary.heading_offset = 0.0
        self.positionTrace = {"own": [], "adversary": []}  # (north, east) per sim step
        self._record_positions()
        self.closest_distance = self._get_current_distance()

    def reward(self, # Reward function.
        p,  # Transition probability
        e,  # An episode accured (the ships collided)
        d,  # Closest distance between the ships throughout the simulation [m]
        terminal):  # Simulation has terminated
        if terminal:
            if e:
                return self.collision_reward
            else:
                return -d/self.miss_distance_scale
        else:
            return math.log(p)

    def plot(self) -> None:
        fig, ax = plt.subplots()
        self.map_gdfs[2].plot(ax=ax, color="#D8D2C4", zorder=0)  # Land
        for name, ship, color in (("Own ship", self.ownShip, "tab:blue"), ("Adversary ship", self.adversary, "tab:red")):
            trace = np.array(self.positionTrace["own" if ship is self.ownShip else "adversary"])
            navigate = ship.auto_pilot.navigate
            ax.plot(navigate.east, navigate.north, color=color, linestyle=":", linewidth=1, label=f'{name} route')
            ax.plot(trace[:, 1], trace[:, 0], color=color, label=f'{name} path')
            ax.scatter(trace[0, 1], trace[0, 0], color=color, marker="o", zorder=5)
            ax.scatter(trace[-1, 1], trace[-1, 0], color=color, marker="x", zorder=5)
        own, adversary = np.array(self.positionTrace["own"]), np.array(self.positionTrace["adversary"])
        closest = int(np.argmin(np.hypot(*(own - adversary).T)))
        ax.plot([own[closest, 1], adversary[closest, 1]], [own[closest, 0], adversary[closest, 0]],
                color="black", marker=".", label=f'Closest distance ({round(self.closest_distance)} m)')
        allPoints = np.vstack([own, adversary])
        margin = 1500
        ax.set_xlim(allPoints[:, 1].min() - margin, allPoints[:, 1].max() + margin)
        ax.set_ylim(allPoints[:, 0].min() - margin, allPoints[:, 0].max() + margin)
        ax.set_aspect("equal")
        ax.set_title(f'Ship paths (o start, x end). Ended by: {self.terminal_reason}')
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.legend()
        plt.show()

    def _check_endstate(self) -> bool:
        distance = self._get_current_distance()
        if distance < self.closest_distance:
            self.closest_distance = distance
        if self.closest_distance < self.collision_distance or "collision" in self._stop_reasons(self.ownShip):
            self.collision_happened = True
            self.terminal_reason = "collision"
        elif self.ownShip.stop or self.adversary.stop:  # Grounding, navigation failure, reached endpoint...
            stopped = self.ownShip if self.ownShip.stop else self.adversary
            self.terminal_reason = f'{stopped.name_tag}: {", ".join(self._stop_reasons(stopped))}'
        elif distance > self.closest_distance + self.pass_margin:
            self.terminal_reason = "ships have passed each other"
        elif self.ownShip.int.time >= self.max_sim_time:
            self.terminal_reason = "max simulation time"
        self.sim_in_endstate = self.terminal_reason is not None
        return self.sim_in_endstate

    def _stop_reasons(self, ship) -> List:  # MAR-AST stores a flag as either bool or [bool, detail]
        return [key for key, value in ship.stop_info.items() if (value[0] if isinstance(value, list) else value)]

    def _record_positions(self) -> None:
        self.positionTrace["own"].append((float(self.ownShip.north), float(self.ownShip.east)))
        self.positionTrace["adversary"].append((float(self.adversary.north), float(self.adversary.east)))

    def _get_current_distance(self) -> float:
        return math.hypot(self.ownShip.north - self.adversary.north, self.ownShip.east - self.adversary.east)

    def _get_transition_probability(self, actionSeed):
        if self.transition_model == "gaussian":  # Following the route (no offset) is the most likely behaviour
            offset = self._get_action_from_seed(actionSeed)
            return math.exp(-offset**2/(2*self.offsetSigma**2))
        if self.lastActionSeed is None:  # "smooth": Same as SimInterface, penalizes change between actions
            return 1
        return 1-abs(actionSeed - self.lastActionSeed)

    def _get_action_from_seed(self, actionSeed):  # Heading offset [rad]
        return (2*actionSeed - 1) * self.maxOffset
