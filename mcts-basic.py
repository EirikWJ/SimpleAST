from mcts.mctsController import MCTSController
from sim.marastInterface import MarAstSimInterface;

MCTSController(simFactory=MarAstSimInterface).loop(50)