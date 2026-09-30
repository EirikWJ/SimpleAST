import math, json, copy

import matplotlib.pyplot as plt

#TODO: Messy simulation. Should be cleaned up.
#
#   Function explenation
#  execute_action(self, action:float) -> p, e, d (transition probability from last action, episode happened, shortest distance throughout sim.)
#   Takes a seed (random float from [0,1]), transfomrs seed into doable action, and executes that aciton on the simulation
#
#  is_endstate(self) -> bool (is the simulation in an endstate?)
#   If boates are within crahs distance or the boat going straight has exited frame collision_happened and sim_in_endstate is updated and sim_in_endstate is returned
#
#  get_state(self) -> action_trace : list[float]
#
#  reset_sim(self) -> None:
#   Sim is reset to start state, variables and everything reset.
#
#  plot(self) -> None:
#   Plots the route of the current state.

class SimpleBoatController:

    def __init__(self) -> None:
        with open("parameters.json") as f:
            params = json.load(f)
        self.steplength = params["steplength"]
        self.crash_distance_threshold = params["crash_distance_threshold"]
        self.action_range = params["action_range"]
        self.reset_sim()
    

    def reset_sim(self):
        with open("parameters.json") as f:
            params = json.load(f)
        self.straight_pos = params["straight_pos"]
        self.steerable_pos = params["steerable_pos"]
        self.steerable_angle = 0
        self.collision_happened = False
        self.sim_in_endstate = False
        self.closest_boat_distance = self._get_current_distance()
            
    def execute_action(self, action: float):  # Action should be scaled by actionIntefrace.
        self.steerable_angle += action*math.pi/100
        self._next_state()
        self.is_endstate()
        e = self.collision_happened
        d = self.closest_boat_distance
        return(e, d)

    def is_endstate(self):
        if self.sim_in_endstate != True:
            if self.closest_boat_distance < self.crash_distance_threshold:
                self.collision_happened = True
                self.sim_in_endstate = True
            elif self.straight_pos[0] > 100:
                self.sim_in_endstate = True
        return self.sim_in_endstate
    
    def plot(self, action_trace):
        steerable_pos_trace, straight_pos_trace = self._get_position_trace(action_trace)
        cdt = self.crash_distance_threshold
        colors = {8*cdt: "gray", 4*cdt: "yellow", 2*cdt: "red", cdt: "black"}
        for i in range(len(steerable_pos_trace)):
            steerable_pos = steerable_pos_trace[i]
            straight_pos = straight_pos_trace[i]
            distance = self._boat_distance(steerable_pos, straight_pos)
            color = "blue"
            for key, _color in colors.items():
                if distance > key:
                    color = _color
                    break
            plt.plot([steerable_pos[0], straight_pos[0]], [steerable_pos[1], straight_pos[1]], c = "#F0F0F0", zorder=0)
            plt.scatter(steerable_pos[0], steerable_pos[1], c = color, zorder=10)
            plt.scatter(straight_pos[0], straight_pos[1], c = color, zorder=10)
        plt.ylim(-10, 110)
        plt.xlim(-10, 110)
        plt.show()

    def _next_state(self):
        self.steerable_pos[0] += math.sin(self.steerable_angle)*self.steplength
        self.steerable_pos[1] += math.cos(self.steerable_angle)*self.steplength
        self.straight_pos[0] += self.steplength
        self._update_closest_distance()
    
    def _get_current_distance(self):
        return self._boat_distance(self.steerable_pos, self.straight_pos)

    def _boat_distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def _update_closest_distance(self):
        distance = self._get_current_distance()
        if distance < self.closest_boat_distance:
            self.closest_boat_distance = distance

    def _get_position_trace(self, action_trace):  # Prev pos is not stored, so sim is reset, and fast forwarded through
        self.reset_sim()
        steerable_state = []
        straight_state = []
        for action in action_trace:
            steerable_state.append(copy.copy(self.steerable_pos))
            straight_state.append(copy.copy(self.straight_pos))
            self.execute_action(action)
        steerable_state.append(copy.copy(self.steerable_pos))
        straight_state.append(copy.copy(self.straight_pos))
        return steerable_state, straight_state
