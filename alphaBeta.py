from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
from open_spiel.python.algorithms import minimax
import numpy as np

class AlphaBetaBot(pyspiel.Bot):
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

    def player_id(self):
        return self._player_id

    def evaluate(self, state, maximizing_player_id):
        """Returns evaluation on given state."""
        state_str = str(state) # state.information_state_string(maximizing_player_id)
        score = [0, 0]
        # 0's pieces 
        for piece, value in [("M", 60), ("C", 1), ("K", 9), ("L", 10), ("J", 8), ("I", 7), ("F", 4), ("G", 5), ("H", 6), ("E", 3), ("B", 5), ("D", 2), ("?", 5)]:
            score[0] += state_str.count(piece)*value
        # 1's pieces
        for piece, value in [("Y", 60), ("O", 1), ("W", 9), ("X", 10), ("V", 8), ("U", 7), ("R", 4), ("S", 5), ("T", 6), ("Q", 3), ("N", 5), ("P", 2), ("?", 5)]:
            score[1] += state_str.count(piece)*value
        # Todo: make more weight to moving forward
        # Todo: make impossible to redo action alternatively
        return (score[maximizing_player_id] - score[1-maximizing_player_id])/120


    def alpha_beta(self, state, depth, value_function, maximizing_player_id):
        """Runs expectiminimax until the specified depth.

        See https://en.wikipedia.org/wiki/Expectiminimax for details.

        Arguments:
            state: The state to start the search from.
            depth: The depth of the search (not counting chance nodes).
            value_function: A value function, taking in a state and returning a value,
            in terms of the maximizing_player_id.
            maximizing_player_id: The player running the search (current player at root
            of the search tree).

        Returns:
            A tuple (value, best_action) representing the value to the maximizing player
            and the best action that achieves that value. None is returned as the best
            action at chance nodes, the depth limit, and terminals.
        """
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
                if child_value > value:
                    value = child_value
                    best_action = action
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

    def step_with_policy(self, state):
        """Returns the stochastic policy and selected action in the given state.

        Args:
        state: The current state of the game.

        Returns:
        A `(policy, action)` pair, where policy is a `list` of
        `(action, probability)` pairs for each legal action, with
        `probability = 1/num_actions`
        The `action` is selected uniformly at random from the legal actions,
        or `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return [], pyspiel.INVALID_ACTION
        policy = [(action, 1/len(legal_actions)) for action in legal_actions]
        value, action = self.alpha_beta(state, 1, self.evaluate, state.current_player())
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]