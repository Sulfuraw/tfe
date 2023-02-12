from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
from statework import stateIntoCharMatrix, print_board, printCharMatrix, flag_protec, generate_possibilities_matrix, is_valid_state, generate_state

# For training purposes only
def everythingEverywhereAllAtOnce(filename, iterations):
    try:
        bot = rnadBot.rnadBot().getSavedState(filename)
        print("Bot loaded by getting", filename)
    except:
        bot = rnadBot.rnadBot()
        print("Bot failed loading by getting", filename)
    bot.train(iterations)
    bot.saveState(filename)
    print("Bot saved successfully")
    print("===========================================")

def saveGame(filename, data):
    with open(filename, 'wb') as outp: # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def getGame(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(None)  # Seed for random
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "ab":
        return alphaBeta.AlphaBetaBot(player_id, game)
    if bot_type == "custom":
        return customBot.CustomBot(game, 1.5, 2000, customBot.CustomEvaluator(), player_id) # customBot.RandomRolloutEvaluator()
    if bot_type == "rnad1":
        try:
            bot = rnadBot.rnadBot().getSavedState("states/state5000.pkl")
            print("Bot rnad 5000 loaded")
            return bot
        except:
            bot = rnadBot.rnadBot()
            print("Bot rnad 5000 failed so start a base bot")
            return bot
    if bot_type == "rnad2":
        try:
            bot = rnadBot.rnadBot().getSavedState("states/state100.pkl")
            print("Bot rnad 100 loaded")
            return bot
        except:
            bot = rnadBot.rnadBot()
            print("Bot rnad 100 failed so start a base bot")
            return bot
    raise ValueError("Invalid bot type: %s" % bot_type)

def _get_action(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None

def _play_game(game, bots):
    """Plays one game."""
    # "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQPNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0"  # Base state debugged
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    history = []
    allStates = []

    for i, bot in enumerate(bots):
        if str(bot) == "custom":
            bot.init_knowledge(state)

    move = 0
    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]

        if str(bot) == "custom":
            # print("\n==============================================")
            # print(state)
            # print(state.information_state_string(state.current_player()))
            generated = generate_state(state, bot.matrix_of_possibilities, bot.information)
            action = bot.step(game.new_initial_state(generated))
            # print(generated)
            # print(is_valid_state(generated))
        else:
            action = bot.step(state)

        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
            if str(bot) == "custom":
                bot.update_knowledge(state.clone(), action)
        history.append(action_str)
        allStates.append(stateIntoCharMatrix(state))

        # print("moves:", move)
        # printCharMatrix(stateIntoCharMatrix(state))
        move+=1

        state.apply_action(action)

    # Game is now done. Print return for each player
    returns = state.returns()
    # print("Game actions:", " ".join(history), "\nReturns:", 
    #         " ".join(map(str, returns)), "\n# moves:", len(history), "\n")
    print("\nReturns:", 
        " ".join(map(str, returns)), "\n# moves:", len(history))
    for bot in bots:
        bot.restart()
    return returns, history, allStates, move

def main(argv):
    game = pyspiel.load_game("yorktown")
    bots = [
        _init_bot(player1, game, 0),
        _init_bot(player2, game, 1),
    ]
    histories = collections.defaultdict(int)
    overall_returns = [0, 0]
    overall_wins = [0, 0]
    game_num = 0
    moves = 0
    try:
        for game_num in range(num_games):
            returns, history, allStates, move = _play_game(game, bots)
            print("Game Number:", game_num)
            if replay:
                wrapper(print_board, allStates, bots, auto)
            saveGame("games/"+str(game_num), allStates)
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
            moves += move
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Average number of moves till finish:", moves//game_num)
    print("Players:", player1, player2)
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)

player1 = "custom"
player2 = "random"
num_games = 50
replay = False
auto = False

if __name__ == "__main__":
    app.run(main)
    # wrapper(print_board, getGame("FullKnown1/30"), ["custom", "random"], auto)
    # wrapper(print_board, getGame("FullKnown1/14lose"), ["custom", "random"], auto)
    # wrapper(print_board, getGame("games/0"), ["custom", "random"], auto)
