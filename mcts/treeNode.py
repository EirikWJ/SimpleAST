import math, json
from typing import Dict

class TreeNode:

    def __init__(self, state):
        self.state = state
        self.timesVisited = 1  # Initiates only after visit so should be one
        self.childrenVisits : Dict[tuple, int] = {}  # ChildNodeState, nrOfVisits
        self.childrenEvaluations : Dict[tuple, tuple] = {}
        with open("parameters.json") as f:
            params = json.load(f)  # Pass out to main?
        self.explorationCoefficient = params["exploration_coefficient"]
    
    def visit_node(self):
        self.timesVisited += 1
    
    def add_child(self, childNodeState):
        childTuple = tuple(childNodeState)
        self.childrenVisits[childTuple] = 0
        self.childrenEvaluations[childTuple] = 0
    
    def visit_child(self, childNodeState):
        self.childrenVisits[tuple(childNodeState)] += 1
    
    def evaluate_child(self, childNodeState, evaluation):
        childTuple = tuple(childNodeState)
        self.childrenEvaluations[childTuple] = self.childrenEvaluations[childTuple] + (evaluation - self.childrenEvaluations[childTuple])/self.childrenVisits[childTuple]

    def backpropegate(self, reward, targetNode):
        if self == targetNode:
            pass
        else:  # TODO: Unsusre if this is correct. Must find out how policy works
            self.totalEvaluation += reward
            self.timesVisited += 1
            if self.parrent is not None:
                self.parrent.backpropegate(reward + self.reward, targetNode)  # TODO: Do we want to return this? Don't see why, but i'm partialy blind.

    def UCTselect(self):  # Tree policy. "Bandit based Monte-Carlo Planning", Kocsis and Szepervari (2006)
        maxValue = -math.inf
        bestChild = None
        for childNodeState in self.childrenVisits:
            x = self.childrenEvaluations[childNodeState] + self.explorationCoefficient*(math.sqrt(math.log(self.timesVisited)/(1+self.childrenVisits[childNodeState])))
            if x > maxValue:
                bestChild = childNodeState
                maxValue = x
        return bestChild