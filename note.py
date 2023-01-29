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
from statework import stateIntoCharMatrix, print_board, printCharMatrix

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

def gameTesting():
    game = pyspiel.load_game("yorktown")
    state = game.new_initial_state()
    print("Initial state:\n{}".format(state))
    for i in range(10):
        print(state.legal_actions())
        print(state.legal_actions()[0], state.action_to_string(state.current_player(), state.legal_actions()[0]))
        state.apply_action(state.legal_actions()[0])
    print(state)
    print(state.information_state_string())
    state.apply_action(state.legal_actions()[0])
    print(state.information_state_string())
    print(state.history())
    # Seems unusable
    # print(state.information_state_tensor())
    # Not implemented
    # print(state.observation_string())
    # print(state.observation_tensor())

def evaluatebot():
    # solver = rnad.RNaDSolver(rnad.RNaDConfig(game_name="yorktown"))
    # result = pyspiel.evaluate_bots(game.new_initial_state(), bots, seed=0)
    return 

################################################################################################
# Old Ones
def _get_action(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None

def _play_game(game, bots, initial_actions):
    """Plays one game."""
    allStates = []
    # "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQPNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0"  # Base state debugged
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    allStates.append(stateIntoCharMatrix(state))
    history = []
    for action_str in initial_actions:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))
        history.append(action_str)
        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)
        allStates.append(stateIntoCharMatrix(state))

    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]
        action = bot.step(state)
        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)
        allStates.append(stateIntoCharMatrix(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))
    print("Number of moves played:", len(history))
    for bot in bots:
        bot.restart()
    return returns, history, allStates

#########################################################################################

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
        # wrapper(main.print_board, [main.stateIntoCharMatrix(state)], ["custom", "random"])
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

player1 = "custom"
player2 = "random"
num_games = 50
replay = False
auto = False

def basic_test():
    game = pyspiel.load_game("yorktown")
    bots = [
        _init_bot(player1, game, 0),
        _init_bot(player2, game, 1),
    ]
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    history = []
    allStates = []

    #while not state.is_terminal():
    for _ in range(1):
        current_player = state.current_player()
        bot = bots[current_player]
        action = bot.step(state)
        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        allStates.append(stateIntoCharMatrix(state))
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

basic_test()



