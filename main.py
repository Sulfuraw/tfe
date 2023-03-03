from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import collections
import time
import pickle
import os.path
from curses import wrapper
from absl import app
import numpy as np
import pandas as pd
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import alphaBeta
import rnadBot
import customBot
from statework import *

def everythingEverywhereAllAtOnce(filename, iterations):
    """Training / actualizing saved weight of the rnadBot"""
    start = time.time()
    try:
        bot = rnadBot.rnadBot().getSavedState(filename)
        print("Bot loaded by getting", filename)
    except:
        bot = rnadBot.rnadBot()
        print("Bot failed loading by getting", filename)
    bot.train(iterations)
    bot.saveState(filename)
    print("Bot saved successfully")
    print("time taken:", time.time() - start)
    print("===========================================")

def saveGame(filename, data):
    """Save the list of state of a game in a file"""
    with open(filename, 'wb') as outp: # Overwrites
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def getGame(filename):
    """Get back the same list of state"""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def save_to_csv(path, data):
    """Save stats in a file / append stats to this file"""
    df = pd.DataFrame(data)
    if os.path.isfile(path):
        df.to_csv(path, mode='a', index=True, header=False)
    else:
        df.to_csv(path, index=True, header=True)

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(None)  # Seed
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "ab":
        return alphaBeta.AlphaBetaBot(player_id, game)
    if bot_type == "customBot":
        return customBot.CustomBot(game, 1.5, 100, customBot.CustomEvaluator(), player_id) # customBot.RandomRolloutEvaluator()
    if bot_type == "rnadBot":
        try:
            bot = rnadBot.rnadBot().getSavedState("states/state5000.pkl")
            print("Bot rnad 5000 loaded")
            return bot
        except:
            bot = rnadBot.rnadBot()
            print("Bot rnad 5000 failed so start a base bot")
            return bot
    raise ValueError("Invalid bot type: %s" % bot_type)

def _play_game(game, bots, game_num):
    """Plays one game."""
    # "FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AASTQPNSQPTSUPWPVRPXPURNQONNQSNVPTNQRRTYUP r 0"  # Base state debugged
    state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    history = []
    allStates = []

    for i, bot in enumerate(bots):
        if str(bot) == "customBot":
            bot.init_knowledge()

    move = 0
    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]

        if str(bot) == "customBot":
            # We adapt the max_simulation parameter to the advancement of the game: 2000 -> 1.0/step to 2s/step, grosse moyenne à 1.3s/step
            if move == 0 or move==1:
                bot.set_max_simulations(100)
            if move == 50 or move == 51:
                bot.set_max_simulations(2000)

            start = time.time()
            generated = generate_state(state, bot.information)
            # Test the generate time for anomaly
            if time.time()-start > 2:
                print("Time for generate was", round(time.time()-start, 2))
            action = bot.step(game.new_initial_state(generated))
            
            # TODO: Pertinent d'avoir le nombre de "?" ou il faudrait le nombre de piece en face qui 
            # reste plutot ? Pour analyse, sum des ? et des pieces ennemies
            # Et rajouter le nombre de piece à nous qu'il reste aussi
            # peut etre même la fonction evaluate sur le vrai state ou le partial state peut être interessant ?????
            data = {'move': [move], 'know_acc': [round(compare_state(state, generated), 2)], 'unk_pieces': [str(state.information_state_string(state.current_player())).count("?")]}
            save_to_csv("./games/"+str(game_num)+".csv", data)
        else:
            action = bot.step(state)

        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
            if str(bot) == "customBot":
                bot.update_knowledge(state.clone(), action)

        history.append(action_str)
        allStates.append(stateIntoCharMatrix(state))
        move+=1
        state.apply_action(action)

    # Game is now done
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
    start_time = time.time()
    try:
        for game_num in range(0, num_games):
            start_game_time = time.time()
            returns, history, allStates, move = _play_game(game, bots, game_num)
            time_taken = round(time.time()-start_game_time)//60
            print("Time for this game in minute:", time_taken)
            print("Game Number:", game_num)

            saveGame("games/"+str(game_num), allStates)
            data = {'game_num': [game_num], 'time_taken': [time_taken], 'moves': move, 'win': returns[0]}
            save_to_csv("./games/stats"+".csv", data)
            
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
            moves += move
            if replay:
                wrapper(print_board, allStates, bots, auto)
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Average number of moves till finish:", moves//(game_num+1))
    print("Average time till finish:", (round(time.time()-start_time)//60)//(game_num+1))
    print("Players:", player1, player2)
    print("Overall wins", overall_wins)

player1 = "customBot"
player2 = "random"
num_games = 10
replay = False
auto = False

if __name__ == "__main__":
    # app.run(main) 
    # wrapper(print_board, getGame("Custom1/3"), ["customBot", "random"], auto) # 271 moves
    # wrapper(print_board, getGame("Custom1/14"), ["customBot", "random"], auto) # 1256 moves, lose
    # wrapper(print_board, getGame("FullKnown1/30"), ["customBot", "random"], auto)
    # wrapper(print_board, getGame("FullKnown1/14lose"), ["customBot", "random"], auto)

    # wrapper(print_board, getGame("Custom3/9"), ["customBot", "random"], auto) #(double prior)
    # wrapper(print_board, getGame("Custom4/4"), ["customBot", "random"], auto) # (is_toward_flag only), 3 et 4
    # wrapper(print_board, getGame("Custom5/1"), ["customBot", "random"], auto) # (toward_flag modified only), 0 et 1
    # wrapper(print_board, getGame("Custom6/9"), ["customBot", "random"], auto) # (re que 2000 en max_iteration et toward_flag), 9 et 5

    # wrapper(print_board, getGame("games/0"), ["customBot", "random"], auto)

    everythingEverywhereAllAtOnce("states/state.pkl", 500) # 14000, 3sec/step
    
