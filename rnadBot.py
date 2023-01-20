from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
from rnad import rnad
import numpy as np
import pickle5 as pickle # Save state

class rnadBot(pyspiel.Bot):
    # A state in str:
    # FEBMBEFEEF
    # BGIBHIBEDB
    # GJDDDHCGJG
    # DHDLIFKDDH
    # AA__AA__AA
    # AA__AA__AA
    # STQQNSQPTS
    # UPWPVRPXPU
    # RNQONNQSNV
    # PTNQRRTYUP r 0

    def __init__(self):
        """Initializes a uniform-random bot.

        Args:
        player_id: The integer id of the player for this bot, e.g. `0` if acting
            as the first player.
        """
        pyspiel.Bot.__init__(self)
        self.solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="yorktown"))

    def __str__(self):
        return "rnadBot"

    def getSolver(self):
        return self.solver

    def setState(self, weights):
        return self.solver.__setstate__(weights)

    def getState(self):
        return self.solver.__getstate__()

    def saveState(self, file):
        weights = self.solver.__getstate__()
        with open(file, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(weights, outp, pickle.HIGHEST_PROTOCOL)

    def getSavedState(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        self.solver.__setstate__(data)
        return self
    
    def train(self, train_steps):
        for i in range(train_steps):
            # if (i % 10 == 0):
            #     print(i, "iteration of training are already done !")
            self.solver.step()
        print("Training Done")
 
    def step_with_policy(self, state):
        policy = ""
        action = self.solver.play_a_move(state)
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]