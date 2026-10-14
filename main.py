from sim.simInterface import SimInterface
from sim.simpleBoatController import SimpleBoatController
from mcts.mctsController import MCTSController
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)
import random

def main():
    MCTS = MCTSController()
    MCTS.loop(4000)

def mainMarast(numberOfLoops = 500):  # Same search, but on MAR-AST's Ship in Transit simulator (parameters_marast.json)
    from sim.marastInterface import MarAstSimInterface
    MCTS = MCTSController(simFactory=MarAstSimInterface)
    MCTS.loop(numberOfLoops)

def mainSac(numberOfLoops = 2000):  # SAC (deep RL) solver instead of MCTS, on the simple boat simulator
    from sac.sacController import SACController
    SAC = SACController()
    SAC.loop(numberOfLoops)

def mainSacMarast(numberOfLoops = 500):  # SAC solver on MAR-AST's Ship in Transit simulator
    from sac.sacController import SACController
    from sim.marastInterface import MarAstSimInterface
    SAC = SACController(simFactory=MarAstSimInterface)
    SAC.loop(numberOfLoops)

def randomPath(verbose = True):
    SBC = SimInterface()
    while not SBC.sim_in_endstate:
        SBC.step(random.random())
    if verbose:
        SBC.plot()
    return SBC.collision_happened

main()

# mainMarast()

# mainSac()

# mainSacMarast()

# randomPath()

# hits = 0
# for i in range(10**6):
#     if randomPath(False):
#         hits+=1
#     if (i+1)%10000 == 0:
#         print(hits, "collisions in",i,"random routes")