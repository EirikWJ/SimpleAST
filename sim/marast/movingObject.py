import math

import numpy as np

class MovingObjectModel:
    ''' A small object (e.g. a swimmer) that moves along a random path of its own. It stands in
        for a ship model in MultiShipEnv, so the ships' SBMPC sees it as an obstacle and MAR-AST's
        collision check includes it.
        The whole path is drawn from np.random.default_rng(seed) when the object is created, so
        the same seed always gives the same path, and different seeds give different paths. The
        path does not react to the ships or to the solver's actions.
    '''

    def __init__(self, center_north, center_east, start_radius, speed, heading_change_sigma,
                 time_step, max_sim_time, seed, name_tag, length, width) -> None:
        rng = np.random.default_rng(seed)
        stepCount = int(math.ceil(max_sim_time/time_step)) + 2
        startDistance = start_radius*math.sqrt(rng.uniform())  # Uniform over the disc around the center
        startDirection = rng.uniform(0, 2*math.pi)
        startNorth = center_north + startDistance*math.cos(startDirection)
        startEast = center_east + startDistance*math.sin(startDirection)
        # Random walk in heading: a random initial heading, then a small random turn every time step
        headingChanges = np.concatenate([[0.0], rng.normal(0, heading_change_sigma, stepCount - 1)])
        self.pathHeading = rng.uniform(0, 2*math.pi) + np.cumsum(headingChanges)
        stepLength = speed*time_step
        self.pathNorth = startNorth + np.concatenate([[0.0], np.cumsum(stepLength*np.cos(self.pathHeading[:-1]))])
        self.pathEast = startEast + np.concatenate([[0.0], np.cumsum(stepLength*np.sin(self.pathHeading[:-1]))])
        self.seed = seed
        self.name_tag = name_tag
        self.l_ship = length
        self.w_ship = width
        self.forward_speed = speed
        self.sideways_speed = 0.0
        self.yaw_rate = 0.0
        self.reset()

    def reset(self) -> None:
        self.pathIndex = 0
        self.stop = False
        self.stop_info = {}
        self._set_pose()

    def step(self, env_args = None, asset_infos = None) -> None:  # Same signature as ShipModel.step()
        self.pathIndex = min(self.pathIndex + 1, len(self.pathNorth) - 1)
        self._set_pose()

    def ship_snap_shot(self) -> None:  # MultiShipEnv calls this when ship drawing is on. Nothing to draw
        pass

    def _set_pose(self) -> None:
        self.north = float(self.pathNorth[self.pathIndex])
        self.east = float(self.pathEast[self.pathIndex])
        self.yaw_angle = float(self.pathHeading[self.pathIndex])
