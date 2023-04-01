from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import numpy as np

# pieces = [["M", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
#           ["Y", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]]
# # It is:  [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
# # Nbr     [1,   6,    1,   8,   5,   4,   4,   4,   3,   2,   1,   1]
# {"M":[0, 0], "B":[1, 0], "C":[2, 0], "D":[3, 0], "E":[4, 0], "F":[5, 0], "G":[6, 0], "H":[7, 0], "I":[8, 0], "J":[9, 0], "K":[10, 0], "L":[11, 0],
#  "Y":[0, 1], "N":[1, 1], "O":[2, 1], "P":[3, 1], "Q":[4, 1], "R":[5, 1], "S":[6, 1], "T":[7, 1], "U":[8, 1], "V":[9, 1], "W":[10, 1], "X":[11, 1]}

class basicAIBot(pyspiel.Bot):
    def __init__(self, player):
        pyspiel.Bot.__init__(self)
        self.player = player
        dicts_of_value = []
        dicts_of_value.append({'B':6,'L':1,'K':1,'J':2,'I':3,'H':4,'G':4,'F':4,'E':5,'D':8,'C':1,'M':1})
        dicts_of_value.append({'N':6,'X':1,'W':1,'V':2,'U':3,'T':4,'S':4,'R':4,'Q':5,'P':8,'O':1,'Y':1})
        #   Bo,   Ms    Ge    Co    Mj    Cp    Lt    Sg    Mi    Sc    Sp    Fl
        # {'B':6,'1':1,'2':1,'3':2,'4':3,'5':4,'6':4,'7':4,'8':5,'9':8,'s':1,'F':1}
        self.total_allies = dicts_of_value[player].copy()
        self.total_ennemies = dicts_of_value[1-player].copy()
        self.total_hidden_allies = dicts_of_value[player].copy()
        self.total_hidden_ennemies = dicts_of_value[1-player].copy()
        self.last_moves = [None, None, None]

        #              Bo, Ms  Ge  Co  Mj  Cp  Lt  Sg  Mi  Sc  Sp  Fl
        self.ranks = [['B','L','K','J','I','H','G','F','E','D','C','M'],
                      ['N','X','W','V','U','T','S','R','Q','P','O','Y']]

    def __str__(self):
        return "basic"
    
    def valuedRank(self, rank, player):
        if self.ranks[player].count(rank) > 0:
            return len(self.ranks[player]) - 2 - self.ranks[player].index(rank)
        else:
            return 0
    
    def interpret_result(): # update knowledge
        pass
    
    def make_move(self, policy, state):
        """Random move is chosen, overwrite this function to implement more bots"""
        actions, proba = np.array(policy).T
        actions = actions.astype(int)
        action = np.random.choice(actions, p=proba)
        return action
    
    def prior(self, state):
        legal_actions = state.legal_actions(state.current_player())
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]
    
    def step_with_policy(self, state):
        policy = self.prior(state)
        action = self.make_move(policy, state)
        self.last_moves = [state.action_to_string(action), self.last_moves[0], self.last_moves[1]]
        return policy, action
    
    def step(self, state):
        return self.step_with_policy(state)[1]