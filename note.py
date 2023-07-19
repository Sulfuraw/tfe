from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel
from absl import app
import numpy as np
from curses import wrapper
from asmodeusBot import *
from customBot import *
from statework import *
import pandas as pd
import heapq
import random

def game_list(_):
    games_list = pyspiel.registered_games()
    print("Registered games:")
    for game in games_list:
        print(" ", game.short_name)
    print()

def count_repetitive_plays_in_history(history):
    hashmap = {}
    for i in history.split(" "):
        prec_value = 0 if hashmap.get(i) == None else hashmap.get(i)
        hashmap[i] = prec_value + 1
    print(hashmap.values())

def example_api_of_games(_):
    print(pyspiel.registered_names())
    game = pyspiel.load_game("tic_tac_toe")
    # Print name of the game: tic_tac_toe()
    print(game)
    # Print max and min utility possible
    print(game.max_utility(), game.min_utility())
    # Print max number of actions
    print(game.num_distinct_actions())

    # Create a state of the game
    state = game.new_initial_state()
    # Print it to see visually
    print(state)
    # Who's the player playing at this state
    print(state.current_player())
    # Is the state a terminal one ? One winner
    print(state.is_terminal())
    # Accumulated reward to all players currently
    print(state.returns())
    # All possible actions: Between 0 and num_distinct_actions()-1
    print(state.legal_actions())
    # apply the action on the state
    state.apply_action(1)

    print("Other Game")
    # Can parametrize the game:
    game = pyspiel.load_game("breakthrough(rows=6,columns=6)")
    # Create a state of the game
    state = game.new_initial_state()
    print(state)
    for action in state.legal_actions():
        print("{} {}".format(action, state.action_to_string(action)))
    state.apply_action(112)
    print(state)

    print("Other games")
    # Create the game matching pennies, automatically generated and same api
    game = pyspiel.create_matrix_game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    state = game.new_initial_state()
    print(state)
    # This will print -2 because it is in simultaneous play
    state.current_player()
    # action of player 0
    state.legal_actions(0)
    # Apply TWO actions bc simultaneous: After that, it's terminal and returns are [1, -1]
    state.apply_actions([0, 0])

    print("Other example")
    # Dynamics API
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_matrix = game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_matrix, dynamics.replicator)
    # Define what the population playes generally with proba
    x = np.array([0.2, 0.2, 0.6])
    # The gradient showed in video
    print(dyn(x))
    # Define a step functions
    alpha = 0.01
    # Manually run
    x += alpha * dyn(x)
    # Multiple steps
    for i in range(10):
        x += alpha * dyn(x)
    print(x)
    return "example finished"

###################################################################################

# Alpha-beta bot, first thing didn't work:
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
        self.history = ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e", "a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]

    def __str__(self):
        return "alphaBetaBot"

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
            score[0] += state_str[-40:].count(piece)
        # 1's pieces
        for piece, value in [("Y", 60), ("O", 1), ("W", 9), ("X", 10), ("V", 8), ("U", 7), ("R", 4), ("S", 5), ("T", 6), ("Q", 3), ("N", 5), ("P", 2), ("?", 5)]:
            score[1] += state_str.count(piece)*value
            score[1] += state_str[:40].count(piece)
        return (score[maximizing_player_id] - score[1-maximizing_player_id])


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
        value, action = self.alpha_beta(state, 2, self.evaluate, state.current_player())
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]
###################################################################################

# The customBot before it started looking like something
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
        # wrapper(main.print_board, [main.stateIntoCharMatrix(state)], ["customBot", "random"])
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

###################################################################################

# Base Evaluator
class Evaluator(object):
    """Abstract class representing an evaluation function for a game.

    The evaluation function takes in an intermediate state in the game and returns
    an evaluation of that state, which should correlate with chances of winning
    the game. It returns the evaluation from all player's perspectives.
    """

    def evaluate(self, state):
        """Returns evaluation on given state."""
        raise NotImplementedError

    def prior(self, state):
        """Returns a probability for each legal action in the given state."""
        raise NotImplementedError

class RandomRolloutEvaluator(Evaluator):
    """A simple evaluator doing random rollouts.

    This evaluator returns the average outcome of playing random actions from the
    given state until the end of the game.  n_rollouts is the number of random
    outcomes to be considered.
    """

    def __init__(self, n_rollouts=5, random_state=None):
        self.n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()

    def evaluate(self, state, id):
        """Returns evaluation on given state."""
        result = None
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            while not working_state.is_terminal():
                action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
            returns = np.array(working_state.returns())
            result = returns if result is None else result + returns

        return result / self.n_rollouts

    def prior(self, state):
        """Returns equal probability for all actions."""
        legal_actions = state.legal_actions(state.current_player())
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

#####################################################################################

# # Debugging asset, printing things for debug
 
# A mettre dans la fonction generate state
                # print("================")
                # printCharMatrix(state.information_state_string(state.current_player()))
                # printCharMatrix(final)
                # print(moved_scout)
                # print(str(state)[i])
                # print(piece_left)
                # print(proba)

# A mettre dans le play game dans le if(str(bot) == "customBot") avec le compare(state, generated)
            # print("\n==============================================")
            # print(state)
            # print(state.information_state_string(state.current_player()))
            # print(compare_state(state, generated))
            # print(generated)
            # print(is_valid_state(generated))

# Pareil dans le play
            # print("\n==============================================")
            # print("move number: " + str(n))
            # printCharMatrix(state)
            # print()
            # printCharMatrix(state.information_state_string(state.current_player()))
            # print("Pieces_left:")
            # print(bot.information[1])
            # print("moved:")
            # print(bot.information[2])
            # print("scout:")
            # print(bot.information[3])
            # print(generated)
            # print(is_valid_state(generated))

###############################################################################

def is_valid_state(state_str):
    """Verify that the state_str is a valid state, to be used for our generate_state"""
    return (state_str.count("M") == 1 
        and state_str.count("Y") == 1
        and 0 <= state_str.count("B") <= 6
        and 0 <= state_str.count("N") <= 6
        and (state_str.count("C") == 1 or state_str.count("C") == 0)
        and (state_str.count("O") == 1 or state_str.count("O") == 0)
        and 0 <= state_str.count("D") <= 8
        and 0 <= state_str.count("P") <= 8
        and 0 <= state_str.count("E") <= 5
        and 0 <= state_str.count("Q") <= 5
        and 0 <= state_str.count("F") <= 4
        and 0 <= state_str.count("R") <= 4
        and 0 <= state_str.count("G") <= 4
        and 0 <= state_str.count("S") <= 4
        and 0 <= state_str.count("H") <= 4
        and 0 <= state_str.count("T") <= 4
        and 0 <= state_str.count("I") <= 3
        and 0 <= state_str.count("U") <= 3
        and 0 <= state_str.count("J") <= 2
        and 0 <= state_str.count("V") <= 2
        and (state_str.count("K") == 1 or state_str.count("K") == 0)
        and (state_str.count("W") == 1 or state_str.count("W") == 0)
        and (state_str.count("L") == 1 or state_str.count("L") == 0)
        and (state_str.count("X") == 1 or state_str.count("X") == 0)
        and state_str.count("?") == 0
        and len(state_str) >= 104)

###############################################################################

# Generate the matrix of possibilites / probabilities used for piece generation in the generate_state
def generate_possibilities_matrix(state, information):
    players_piece = players_pieces
    player_id, nbr_piece_left, moved_before, moved_scout = information
    matrix = stateIntoCharMatrix(state.information_state_string(player_id))
    matrix_of_possibilities = np.zeros((10, 10, 12))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == "?":
                if moved_scout[i][j]:
                    only_one_proba = [0.0]*12
                    only_one_proba[3] = 1.0
                    matrix_of_possibilities[i:i+1, j:j+1] = only_one_proba
                elif moved_before[i][j]:
                    matrix_of_possibilities[i:i+1, j:j+1] = moved_piece(nbr_piece_left)
                else:
                    matrix_of_possibilities[i:i+1, j:j+1] = no_info_piece(nbr_piece_left)
            # Only need information of proba of ennemy
            elif matrix[i][j] in players_piece[1-player_id]:
                for p in range(len(players_piece[1-player_id])):
                    if matrix[i][j] == players_piece[1-player_id][p]:
                        only_one_proba = [0.0]*12
                        only_one_proba[p] = 1.0
                        matrix_of_possibilities[i:i+1, j:j+1] = only_one_proba
    return matrix_of_possibilities

# Generate a valid state, with the knowledge of our bot + the partial state
def generate_state_via_matrix(state, matrix_of_possibilities, information):
    players_piece = players_pieces()
    partial_state = state.information_state_string(state.current_player())
    state_str = str(partial_state)
    piece_left = information[1].copy()
    moved_before = information[2]
    final = ""

    while not is_valid_state(final):
        final = ""
        done = set()
        i = 0
        max_try = 0
        while i < len(state_str):
            if state_str[i] == "?" and max_try < 1000:
                # If the number of possible places for unmoved pieces is near the real number of this type of piece,
                # We need to increase their probability way higher
                if (state_str.count("?") - np.sum(moved_before) < piece_left[0] + piece_left[1] + 2) and matrix_of_possibilities[i//10][i%10][0] > 0:
                    flag = information[1][0]
                    bomb = information[1][1]
                    piece_id = np.random.choice(np.arange(12), p=[flag/(flag+bomb), bomb/(flag+bomb), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    piece_id = np.random.choice(np.arange(12), p=matrix_of_possibilities[i//10][i%10])
                # When a piece has been generated its maxinum number of time don't allow it to regenerate again, we reroll in case it happen
                if piece_id not in done:
                    piece_left[piece_id] -= 1
                    if piece_left[piece_id] == 0:
                        done.add(piece_id)
                    piece = players_piece[1-state.current_player()][piece_id]
                    final += piece
                    i += 1
                else:
                    max_try += 1
            else:
                piece = state_str[i]
                final += piece
                i += 1
    return final

###############################################################################
# 15 avril

def generate_state_old(state, information):
    """
        Generate a valid state, with the knowledge of our bot + the partial state
        Information : [self.player_id, nbr_piece_left, moved_before, moved_scout]
    """
    player = state.current_player()
    partial_state = state.information_state_string(player)
    piece_left = information[1].copy()
    moved_before = information[2]
    moved_scout = information[3]
    players_piece = players_pieces()

    state_list = list(partial_state)

    unk_coord = []
    for i in range(100):
        if state_list[i] == "?":
            unk_coord.append(i)

    if player:
        unk_coord = unk_coord[::]
    
    # Treat the scout and moved piece first
    last_unk_coord = []
    for i in unk_coord:
        if moved_scout[i//10][i%10]:
            proba = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif moved_before[i//10][i%10]:
            proba = moved_piece(piece_left)
        else:
            last_unk_coord.append(i)
            continue
        piece_id = np.random.choice(np.arange(12), p=proba)
        if not moved_scout[i//10][i%10]:
            piece_left[piece_id] -= 1
        piece = players_piece[1-player][piece_id]
        state_list[i] = piece
    
    # Comeback to the ones we didn't treat: the one that never moved
    for i in last_unk_coord:
        proba = no_info_piece(piece_left)
        piece_id = np.random.choice(np.arange(12), p=proba)
        piece_left[piece_id] -= 1
        piece = players_piece[1-player][piece_id]
        state_list[i] = piece
    
    state_str = "".join(state_list)
    return state_str

def updating_knowledge_old(information, state, action):
    """
        Each move a player does can affect the information / knowledge that our bot has
        Here we update the variable information
        Information : [self.player_id, nbr_piece_left, moved_before, moved_scout]
    """
    player_id, nbr_piece_left, moved_before, moved_scout = information
    players_piece = players_pieces()
    current_player = state.current_player()

    coord = action_to_coord(state.action_to_string(current_player, action))

    full_matrix_before = stateIntoCharMatrix(state)
    matrix_before = stateIntoCharMatrix(state.information_state_string(player_id))
    start = matrix_before[coord[1]][coord[0]]
    arrival = matrix_before[coord[3]][coord[2]]

    # This is ok because the state passed in argument was a clone
    state.apply_action(action)
    matrix_after = stateIntoCharMatrix(state.information_state_string(player_id))
    arrival_after = matrix_after[coord[3]][coord[2]]

    if current_player == player_id:
        # Fight
        if arrival == "?":
            # Either win or lose the fight, it will result in the same operation
            # We also delete information of moved piece at this place because it doesnt matter anymore
            for p in range(12):
                if full_matrix_before[coord[3]][coord[2]] == players_piece[1-player_id][p]:
                    if not (p==3 and moved_scout[coord[3]][coord[2]]):
                        nbr_piece_left[p] -= 1
                    moved_before[coord[3]][coord[2]] = 0
                    moved_scout[coord[3]][coord[2]] = 0
    else:
        # Fight
        if arrival in players_piece[player_id]:
            was_scout = moved_scout[coord[1]][coord[0]]
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 0
            moved_scout[coord[1]][coord[0]] = 0
            moved_scout[coord[3]][coord[2]] = 0
            # Ennemy Won: But if he killed our piece, we get information of it
            if arrival_after in players_piece[current_player]:
                if start == "?":
                    for p in range(12):
                        if arrival_after == players_piece[current_player][p]:
                            if not (p==3 and was_scout):
                                nbr_piece_left[p] -= 1
            # Ennemy lose:
            else:
                for p in range(12):
                    if full_matrix_before[coord[1]][coord[0]] == players_piece[current_player][p] and start == "?":
                        if not (p==3 and was_scout):
                            nbr_piece_left[p] -= 1
        # Deplacement on empty space: Deplacement of moved and/or get information about moved/scout_moved
        else:
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 1
            if abs(coord[1]-coord[3]) > 1 or abs(coord[0]-coord[2]) > 1:
                if not moved_scout[coord[1]][coord[0]] and start == "?":
                    nbr_piece_left[3] -= 1
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1
            if moved_scout[coord[1]][coord[0]]:
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1

    # # It is:  [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
    # # Nbr     [1,   6,    1,   8,   5,   4,   4,   4,   3,   2,   1,   1]
    # def value_for_piece_old(self, ennemy_nbr_pieces):
    #     value = np.array([0]*12)
    #     for i in range(3, 12):
    #         value[i] = np.sum(ennemy_nbr_pieces[2:(i+1)])+2
    #     value[0] = 100.0
    #     value[1] = np.sum(ennemy_nbr_pieces[2:]) if ennemy_nbr_pieces[4] == 0 else (np.sum(ennemy_nbr_pieces[2:]) - ennemy_nbr_pieces[4])*0.8
    #     value[2] = 9.0 if ennemy_nbr_pieces[11] else 1.0
    #     value[4] += ennemy_nbr_pieces[1]
    #     value = value / np.max(value[1:])
    #     returns = []
    #     for i in range(12):
    #         returns.append((i, value[i]))
    #     return returns

    # Less old but it was not working well
    # for i in range(4, 12):
    # # This line cause lots of problems:
    # #   I suicide a piece and it gives me more evaluation
    # #   I kill a small piece and it gives me REALLY much value +0.7 quand all miner eliminé ? 
    # #   That is the inverse of what we want...
    # value[i] = value[i-1]*1.6 if enemies[i] > 0 else value[i-1]


    # def evaluate_state(self, state, move_of_state):
    #     """Returns evaluation on given state."""
    #     state_str = str(state)

    #     nbr_pieces = [[0]*12, [0]*12]
    #     for piece in state_str[:100].upper():
    #         if piece not in ["A", "_"]:
    #             i, player = self.piece_to_index[piece]
    #             nbr_pieces[player][i] += 1

    #     score = [0, 0]
    #     for player in [0, 1]:
    #         for piece_id, value in self.value_for_piece(nbr_pieces[1-player]):
    #             score[player] += nbr_pieces[player][piece_id]*value

    #             # make more weight if flag is protected
    #             # score[player] += 1 if flag_protec(state, player) else 0

    #             # # make more weight having pieces on the other part of the board, go forward
    #             if player:
    #                 score[player] += state_str[:40].count(self.player_pieces[player][piece_id])/20
    #             else: 
    #                 score[player] += state_str[-40:].count(self.player_pieces[player][piece_id])/20

            # # Make more weight to miner to go attack in search of bombs
            # if player:
            #     score[player] += 3.0 if self.player_pieces[player][4] in state_str[:50] else 0.0
            # else:
            #     score[player] += 3.0 if self.player_pieces[player][4] in state_str[-50:] else 0.0
    #     returns = [0, 0]
    #     for player in [0, 1]:
    #         returns[player] = score[player] - np.sum(nbr_pieces[1-player])/2   # Re-range with (x - min)/(max-min)
    #     return returns    

    # def is_forward(self, action, player):
    # _, pos1, _, pos2 = list(action)
    # return pos1 < pos2 if player == 0 else pos1 > pos2

    # if arrival != "A":
    #     ally = state_str[coord[1]*10 + coord[0]]
    #     clone = state.clone()
    #     clone.apply_action(action)
    #     clone_str = str(clone).upper()
    #     arrival_after = clone_str[coord[3]*10 + coord[2]]
    #     value = (value + 15) if (ally == arrival_after) else (value/3)

###############################################################################

# Define the A* algorithm function
def astar(array, start, end):
    # Define the heuristic function (Manhattan distance)
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    # Create an empty set to store visited nodes
    visited = set()
    # Create a priority queue to store nodes to be visited
    queue = [(0, start)]
    # Create a dictionary to store the distance from the start node to each node
    distance = {start: 0}
    # Create a dictionary to store the path from the start node to each node
    path = {start: [start]}
    
    # Loop until all nodes have been visited or the end node has been found
    while queue:
        # Get the node with the lowest estimated distance to the end node
        (cost, current) = heapq.heappop(queue)
        
        # Check if the current node is the end node
        if current == end:
            # Return the path to the end node
            return path[current]
        
        # Check if the current node has already been visited
        if current in visited:
            continue
        
        # Mark the current node as visited
        visited.add(current)
        
        # Loop over the neighbors of the current node
        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), (current[0], current[1]+1), (current[0], current[1]-1)]:
            # Check if the neighbor is within the bounds of the array
            if neighbor[0] >= len(array) or neighbor[0] < 0 or neighbor[1] >= len(array[0]) or neighbor[1] < 0:
                continue
            
            # Check if the neighbor is a valid node
            if array[neighbor[0]][neighbor[1]] != 'A' and (neighbor[0], neighbor[1]) != end:
                continue
            
            # Calculate the distance from the start node to the neighbor node
            cost = distance[current] + 1
            
            # Check if the neighbor node has already been visited or if it has a shorter distance from the start node
            if neighbor not in distance or cost < distance[neighbor]:
                # Update the distance and path dictionaries
                distance[neighbor] = cost
                priority = cost + heuristic(end, neighbor)
                heapq.heappush(queue, (priority, neighbor))
                path[neighbor] = path[current] + [neighbor]
    # If the end node was not found, return None
    return None

###############################################################################

# Simulation rollout prior random:
                # # We simulate our moves with prior and simulate ennemy move randomly.
                # if (i%2) == 0:
                #     legal_actions = self.prior(working_state)
                #     actions, proba = list(zip(*legal_actions))

                #     # Transform proba to delete values below a treshold and help converge the results
                #     proba = np.array(proba)
                #     if np.max(proba) > 0.045: #(move_of_state+i) > 100
                #         proba[proba <= 0.045] = 0
                #         proba = proba/np.sum(proba)

                #     action = np.random.choice(actions, p=proba)
                # else:
                #     action = np.random.choice(working_state.legal_actions())

###############################################################################

state = pyspiel.load_game("yorktown").new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0")
nbr_piece_left = np.array([1, 8, 1, 6, 5, 4, 4, 4, 3, 2, 1, 1])
moved_before = np.zeros((10, 10))
moved_scout = np.zeros((10, 10))
information = [0, nbr_piece_left, moved_before, moved_scout, matrix_of_stats(0)]

# state = pyspiel.load_game("yorktown").new_initial_state("FEBBMFFBDFBHAJDCEGDBHAKEDGJEDGBGEAAXAIDHAI__AA__AAAA__AA__AARANAARPSPSNPWPLASUTVRQPQUQONQNQPPNPTTYNR r 58")
# eval = CustomEvaluator()
# eval.set_information(information)
# prio = eval.prior(state)
# for i in range(len(prio)):
#     print("{} -> {}: {}".format(prio[i][0], action_to_coord(state.action_to_string(prio[i][0])), prio[i][1]))

