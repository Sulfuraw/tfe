from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import collections
import random
import sys
import time
import curses
from curses import wrapper
from absl import app
import numpy as np
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import alphaBeta
import rnadBot

player1 = "rnad5000"
player2 = "random"
num_games = 1
replay = True
auto = False

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

def stateIntoCharMatrix(state):
    stat = str(state).split(" ")[0]
    stat = ' '.join(stat)
    statInLines = []
    for i in range(0, len(stat), 20):
        statInLines.append(stat[i:i+20].rstrip().split(" "))
    return statInLines

def print_board(stdscr, states):
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)
    nthState = 0
    while nthState < len(states):
        state = states[nthState]
        for i in range(len(state)):
            for j in range(len(state[i])):
                oldPiece = state[i][j].upper()
                if oldPiece in ["M", "C", "K", "L", "J", "I", "F", "G", "H", "E", "B", "D"]:
                    color = curses.color_pair(1)
                elif oldPiece in ["Y", "O", "W", "X", "V", "U", "R", "S", "T", "Q", "N", "P"]:
                    color = curses.color_pair(4)
                else: color = curses.color_pair(7)
                if oldPiece in ["M", "Y"]:
                    piece = "Fl"
                elif oldPiece in ["C", "O"]:
                    piece = "Sp"
                elif oldPiece in ["D", "P"]:
                    piece = "Sc"
                elif oldPiece in ["E", "Q"]:
                    piece = "Mi"
                elif oldPiece in ["F", "R"]:
                    piece = "Sg"
                elif oldPiece in ["B", "N"]:
                    piece = "Bo"
                elif oldPiece in ["G", "S"]:
                    piece = "Lt"
                elif oldPiece in ["H", "T"]:
                    piece = "Cp"
                elif oldPiece in ["I", "U"]:
                    piece = "Mj"
                elif oldPiece in ["J", "V"]:
                    piece = "Co"
                elif oldPiece in ["K", "W"]:
                    piece = "Ge"
                elif oldPiece in ["L", "X"]:
                    piece = "Ms"
                elif oldPiece in ["_"]:
                    piece = "##"
                elif oldPiece in ["A"]:
                    piece = "__"
                else: piece = "??"
                stdscr.addstr(i+1, j*2+j+2, piece, color)
        if auto:
            stdscr.timeout(1000)
        while True:
            c = stdscr.getch()
            if not auto:
                if c == ord('d'):
                    nthState += 1
                    break
                if c == ord('s'):
                    nthState -= 1
                    break
                if c == ord('f'):
                    nthState += 2500
                    break
            else:
                nthState += 1
                break

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(None)  # Seed for random
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "ab":
        return alphaBeta.AlphaBetaBot(player_id, game)
    if bot_type == "rnad5000":
        try:
            bot = rnadBot.rnadBot().getSavedState("state5000.pkl")
            print("Bot rnad 5000 loaded")
            return bot
        except:
            bot = rnadBot.rnadBot()
            print("Bot rnad 5000 failed so start a base bot")
            return bot
    if bot_type == "rnad100":
        try:
            bot = rnadBot.rnadBot().getSavedState("state100.pkl")
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

def _play_game(game, bots, initial_actions):
    """Plays one game."""
    allStates = []
    # "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQPNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0"  # Base state debugged
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    # wrapper(print_board, stateIntoCharMatrix(state))
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
        # wrapper(print_board, stateIntoCharMatrix(state))
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
        # wrapper(print_board, stateIntoCharMatrix(state))
        allStates.append(stateIntoCharMatrix(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))
    print("Number of moves played:", len(history))
    for bot in bots:
        bot.restart()
    if replay:
        wrapper(print_board, allStates)
    return returns, history

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
    try:
        for game_num in range(num_games):
            returns, history = _play_game(game, bots, argv[1:])
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Number of distinct games played:", len(histories))
    print("Players:", player1, player2)
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)


if __name__ == "__main__":
    app.run(main)







# t1 = time.time()
# everythingEverywhereAllAtOnce("state500.pkl", 500)
# t2 = time.time()
# print(t2-t1)

# t1 = time.time()
# everythingEverywhereAllAtOnce("state1000.pkl", 1000)
# t2 = time.time()
# print(t2-t1)

# t1 = time.time()
# everythingEverywhereAllAtOnce("state5000.pkl", 5000)
# t2 = time.time()
# print(t2-t1)