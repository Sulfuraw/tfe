from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel
import collections
import time
import pickle
from curses import wrapper
from absl import app
import numpy as np
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import alphaBeta
import rnadBot
import customBot
from main import _init_bot
from statework import *
import pandas as pd

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

#########################################################################################

# Alpha-beta bot, first thing didn't work:
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import pyspiel
# from open_spiel.python.algorithms import minimax
# import numpy as np

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
#########################################################################################


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

###############################################################################

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

#####################################################################################################

# Print a graph 
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def graph():
    df = pd.read_csv('games/0.csv')
    x = df["move"]
    y = df["unknow_acc"]
    plt.plot(x, y)
    plt.xlabel("Number of moves")
    plt.ylabel("knowledge accuracy")
    plt.ylim(0.0, 1.0)
    plt.show()

#####################################################################################################

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

# printCharMatrix(state.information_state_string(maximizing_player_id))

# Base state debugged
# "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQPNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0"
# Equal state
# state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0")
# Test avec un miner (E) plus devant seul devant les mines du flag
# state = game.new_initial_state("FDBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHHLIFKDDEAA__AA__AAAA__AA__AAAAAAAAAAAAAAAAAAAAAAAAAAAANUSNRQQRQNYNQR r 205")

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

#More weight moving forward with game advancement
                # for move, adv, div in [(25, 50, 20), (75, 40, 20), (200, 30, 10)]:
                #     if move_of_state > move:
                #         if player: # player = 1
                #             score[player] += state_str[:adv].count(self.player_pieces[player][piece_id])/div
                #         else:
                #             score[player] += state_str[-adv:].count(self.player_pieces[player][piece_id])/div

# Change way of counting with game advancement
                # if move_of_state < 200:
                #     score[player] += nbr_pieces[player][piece_id]*value
                # else:
                #     score[player] += nbr_pieces[player][piece_id]

# Adapt the max_simulation during the game: Useless-sama ?
            # We adapt the max_simulation parameter to the advancement of the game:
            # if move == 0 or move==1:
            #     bot.set_max_simulations(100)
            # if move == 50 or move == 51:
            #     bot.set_max_simulations(2000)

###############################################################################
def basic_test():
    game = pyspiel.load_game("yorktown")
    bots = [
        _init_bot(player1, game, 0),
        _init_bot(player2, game, 1),
    ]
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    
    for i, bot in enumerate(bots):
        if str(bot) == "customBot":
            bot.init_knowledge(state)
    
    history = []
    allStates = []
    allStates.append(stateIntoCharMatrix(state))

    #while not state.is_terminal():
    for n in range(10):
        current_player = state.current_player()
        bot = bots[current_player]

        if str(bot) == "customBot":
            generated = generate_state(state, bot.information)
            action = bot.step(game.new_initial_state(generated))
        else:
            action = bot.step(state)
        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
            if str(bot) == "customBot":
                bot.update_knowledge(state.clone(), action)
        history.append(action_str)
        state.apply_action(action)
        allStates.append(stateIntoCharMatrix(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Game actions:", " ".join(history), "\nReturns:", 
            " ".join(map(str, returns)), "\n# moves:", len(history), "\n")
    for bot in bots:
        bot.restart()
    wrapper(print_board, allStates, bots, auto)
    return returns, history, allStates

player1 = "customBot"
player2 = "random"
num_games = 50
replay = False
auto = False
# basic_test()

pieces = [["M", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
              ["Y", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]]
    # It is:  [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
    # Nbr     [1,   6,    1,   8,   5,   4,   4,   4,   3,   2,   1,   1]

# print(len("FDBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHHLIFKDDEAA__AA__AAAA__AA__AAAAAAAAAAAAAAAAAAAAAAAAAAAENUSNRQQRQNYNQR r 0"))
# print(len("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 1"))

state = pyspiel.load_game("yorktown").new_initial_state("FDBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHHLIFKDDAAA__AA__AAAA__AA__AAAAAEAAAAAAAAAAAAAAAAAAAAAANUSNRQQRQNYNQR r 205")
action = state.legal_actions(state.current_player())[5]
player = state.current_player()
