from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import numpy as np
import main
from curses import wrapper

class CustomBot(pyspiel.Bot):
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

    def __init__(self, player_id, game):
        """Initializes a uniform-random bot.

        Args:
        player_id: The integer id of the player for this bot, e.g. `0` if acting
            as the first player.
        """
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self.game = game
        self._np_rng = np.random.RandomState(33)
        self.history = ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]

    def __str__(self):
        return "customBot"

    def player_id(self):
        return self._player_id

    def evaluate(self, state, maximizing_player_id):
        """Returns evaluation on given state."""
        state_str = str(state) # state.information_state_string(maximizing_player_id)
        score = [0, 0]
        # 0's pieces 
        for piece, value in [("M", 60), ("C", 1), ("K", 9), ("L", 10), ("J", 8), ("I", 7), ("F", 4), ("G", 5), ("H", 6), ("E", 3), ("B", 5), ("D", 2), ("?", 5)]:
            score[0] += state_str.count(piece)*value
            # make more weight to moving forward
            score[0] += state_str[-50:].count(piece)
        # 1's pieces
        for piece, value in [("Y", 60), ("O", 1), ("W", 9), ("X", 10), ("V", 8), ("U", 7), ("R", 4), ("S", 5), ("T", 6), ("Q", 3), ("N", 5), ("P", 2), ("?", 5)]:
            score[1] += state_str.count(piece)*value
            score[1] += state_str[:50].count(piece)
        return (score[maximizing_player_id] - score[1-maximizing_player_id])


    def alpha_beta(self, state, depth, value_function, maximizing_player_id):
        if state.is_terminal():
            return state.player_return(maximizing_player_id), None

        if depth == 0:
            return value_function(state, maximizing_player_id), None

        if state.current_player() == maximizing_player_id:
            value = -float("inf")
            for action in state.legal_actions():
                child_state = state.clone()
                child_state.apply_action(action)
                child_value, _ = self.alpha_beta(child_state, depth - 1, value_function,
                                                maximizing_player_id)
                if child_value > value and action not in self.history[-15:]:
                    value = child_value
                    best_action = action
            # Cannot do multiple time the same action
            self.history.append(best_action)
            return value, best_action
        else:
            value = float("inf")
            for action in state.legal_actions():
                child_state = state.clone()
                child_state.apply_action(action)
                child_value, _ = self.alpha_beta(child_state, depth - 1, value_function,
                                                maximizing_player_id)
                if child_value < value:
                    value = child_value
                    best_action = action
            return value, best_action
    
    def policy_from_actions(self, state):
        policy = []
        minimum_pain = [-1000, 0]
        place=0
        wrapper(main.print_board, [main.stateIntoCharMatrix(state)], ["custom", "random"])
        for action in state.legal_actions():
            child_state = state.clone()
            child_state.apply_action(action)
            child_value = self.evaluate(child_state, self._player_id)
            if (child_value < 0): 
                if (child_value > minimum_pain[0]):
                    minimum_pain = [child_value, place]
                child_value = 0
            policy.append(child_value)
            place += 1
        policy_sum = sum(policy)
        if policy_sum == 0:
            policy_sum = 1
            policy[minimum_pain[1]] = 1
        policy = np.array(policy) / policy_sum
        # print(policy)
        return policy
        
    def step_with_policy(self, state):
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return [], pyspiel.INVALID_ACTION
        policy = self.policy_from_actions(state)
        action = self._np_rng.choice(legal_actions, p=policy)
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]