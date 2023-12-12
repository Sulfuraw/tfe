from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import collections
import time
import pickle
import os.path
from curses import wrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import rnadBot
import customBot
import basicAIBot
import asmodeusBot
import hunterBot
import mctsBot
import gptBot
from statework import *
from script import *

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
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False, header=True)


def play_game_versus_doi(game_num, player1, player2):
    """Plays one game."""
    # Delete t.txt data before starting
    try:
        open("temp/t.txt", 'w').close()
    except IOError:
        print('Failure')
    game = pyspiel.load_game("yorktown")
    bots = [
        _init_bot(player1, game, 0),
        _init_bot(player2, game, 1),
    ]

    player_pieces = players_pieces()
    setups = [
        # [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
        [5, 4, 1, 0, 1, 4, 5, 4, 4, 5, 
         1, 6, 8, 1, 7, 8, 1, 4, 3, 1, 
         6, 9, 3, 3, 3, 7, 2, 6, 9, 6, 
         3, 7, 3, 11, 8, 5, 10, 3, 3, 7], # Base p1
        [3, 8, 0, 7, 5, 5, 4, 1, 7, 3, 
         9, 1, 6, 4, 1, 1, 2, 4, 1, 5, 
         8, 3, 11, 3, 5, 9, 3, 10, 3, 8, 
         6, 7, 3, 4, 6, 1, 3, 4, 7, 6] # Base p2
        ]
    # player_pieces[0] means that it's the red player pieces
    s1T = "".join([player_pieces[0][x] for x in setups[0]])
    s2T = "".join([player_pieces[0][x] for x in setups[1]])
    s1B = [""]*40
    s2B = [""]*40
    for i in range(4):
        for j in range(10):
            s1B[i*10+j] = player_pieces[1][setups[0][(3-i)*10+j]]
            s2B[i*10+j] = player_pieces[1][setups[1][(3-i)*10+j]]
    s1B = "".join(s1B)
    s2B = "".join(s2B)


    setup = s2T + "AA__AA__AAAA__AA__AA" + s1B + " r 0"
    state = game.new_initial_state(setup)
    # wrapper(print_board, [stateIntoCharMatrix(state)], [player1, player2], auto=False)
    history = []
    allStates = []

    for i, bot in enumerate(bots):
        if str(bot) == "custom" or str(bot) == "hunter" or str(bot) == "mcts":
            bot.init_knowledge()

    move = 0
    play = 0
    nbr_of_prior = 0
    per_of_hit = 0
    while not state.is_terminal():
        print("====================== " + ("TOP" if state.current_player() == 0 else "BOTTOM") + " " + str(move) + " ======================\n")
        printCharMatrix(state)
        # Draw by the rules
        if move == 1501: # 2001
            pieces0 = 0
            pieces1 = 0
            players_piece = players_pieces()
            for i in str(state)[:100].upper():
                if i in players_piece[0]:
                    pieces0 += 1
                if i in players_piece[1]:
                    pieces1 += 1
            returns = state.returns()
            print("\nReturns:", " ".join(map(str, returns)), "\n# moves:", len(history))
            for bot in bots:
                if str(bot) == "custom":
                    hash = bot.evaluator.hash_count
                    nbr_of_prior = hash[2]
                    per_of_hit = round((hash[3]/hash[2])*100, 3)
                    print("Hit percentage of hash in prior: ", per_of_hit, nbr_of_prior)
            for bot in bots:
                bot.restart()
            saveGame("games/"+player1+"-"+player2+str(game_num), allStates)
            # In case of tie, both win_1 and win_2 are equal to 0, otherwise the winner is 1, the other is 0
            data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num], 'time_taken': [0], 'moves': [move], 'nbr_of_prior': [nbr_of_prior], 'per_of_hit': [per_of_hit], 'win_1': 1 if returns[0]==1.0 else 0, 'win_2': 1 if returns[1]==1.0 else 0, 'pieces_1': [pieces0], 'pieces_2': [pieces1]}
            save_to_csv("./temp/games/stats.csv", data)
            return returns, history, allStates, move, nbr_of_prior, per_of_hit, [pieces0, pieces1]

        current_player = state.current_player()
        bot = bots[current_player]

        if str(bot) == "custom" or str(bot) == "mcts":
            start = time.time()
            generated = game.new_initial_state(generate_state(state, bot.information))
            # Test the generate time for anomaly
            if time.time()-start > 2:
                print("Time for generate was", round(time.time()-start, 2))
            action = bot.step(generated)
            # print("time for generate and play a move:", time.time()-start)
            if str(bot) == "custom":
                save_to_csv("./temp/games/"+player1+"-"+player2+str(game_num)+".csv", data_for_games(move, state, generated, bot.evaluator))
        elif str(bot) == "hunter":
            action = bot.step(state)
        else:
            while True:
                try:
                    with open(('temp/t.txt'), 'r') as file:
                        lines = file.readlines()
                        if len(lines) > play:
                            break
                except:
                    pass
                time.sleep(1)
            action = bot.step(state)
            play += 1
        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
            if str(bot) == "custom" or str(bot) == "hunter" or str(bot) == "mcts":
                bot.update_knowledge(state.clone(), action)

        history.append(action_str)
        allStates.append(stateIntoCharMatrix(state))
        move+=1
        state.apply_action(action)
        print()
        print(action_str)

    pieces0 = 0
    pieces1 = 0
    players_piece = players_pieces()
    for i in str(state)[:100].upper():
        if i in players_piece[0]:
            pieces0 += 1
        if i in players_piece[1]:
            pieces1 += 1
    # Game is now done
    returns = state.returns()
    print("\nReturns:", 
        " ".join(map(str, returns)), "\n# moves:", len(history))
    for bot in bots:
        if str(bot) == "custom":
            hash = bot.evaluator.hash_count
            nbr_of_prior = hash[2]
            per_of_hit = round((hash[3]/hash[2])*100, 3)
            print("Hit percentage of hash in prior: ", per_of_hit, nbr_of_prior)
    for bot in bots:
        bot.restart()
    saveGame("games/"+player1+"-"+player2+str(game_num), allStates)
    # In case of tie, both win_1 and win_2 are equal to 0, otherwise the winner is 1, the other is 0
    data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num], 'time_taken': [0], 'moves': [move], 'nbr_of_prior': [nbr_of_prior], 'per_of_hit': [per_of_hit], 'win_1': 1 if returns[0]==1.0 else 0, 'win_2': 1 if returns[1]==1.0 else 0, 'pieces_1': [pieces0], 'pieces_2': [pieces1]}
    save_to_csv("./temp/games/stats.csv", data)
    return returns, history, allStates, move, nbr_of_prior, per_of_hit, [pieces0, pieces1]

def _play_game(game, bots, game_num):
    """Plays one game."""
    # state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    state = game.new_initial_state(create_initial_state_str())
    history = []
    allStates = []
    nbr_of_prior = 0
    per_of_hit = 0
    log("Starting a new game!\n" + str(bots[0]) + " vs " + str(bots[1]) + ": Game number "+str(game_num))

    for i, bot in enumerate(bots):
        if str(bot) == "custom" or str(bot) == "hunter" or str(bot) == "mcts":
            bot.init_knowledge()

    move = 0
    while not state.is_terminal():
        # Draw by the rules
        if move == 1500: # 2001
            pieces0 = 0
            pieces1 = 0
            players_piece = players_pieces()
            for i in str(state)[:100].upper():
                if i in players_piece[0]:
                    pieces0 += 1
                if i in players_piece[1]:
                    pieces1 += 1
            returns = state.returns()
            print("\nReturns:", 
                " ".join(map(str, returns)), "\n# moves:", len(history))
            for bot in bots:
                if str(bot) == "custom":
                    hash = bot.evaluator.hash_count
                    nbr_of_prior = hash[2]
                    per_of_hit = round((hash[3]/hash[2])*100, 3)
                    print("Hit percentage of hash in prior: ", per_of_hit, nbr_of_prior)
            for bot in bots:
                bot.restart()
            return returns, history, allStates, move, nbr_of_prior, per_of_hit, [pieces0, pieces1]         

        current_player = state.current_player()
        bot = bots[current_player]

        if str(bot) == "custom" or str(bot) == "mcts":
            start = time.time()
            generated = game.new_initial_state(generate_state(state, bot.information))
            # Test the generate time for anomaly
            if time.time()-start > 2:
                print("Time for generate was", round(time.time()-start, 2))

            log("Move number: "+str(move))
            log("Generated state:\n"+ returnCharMatrix(generated))
            log("Real state:\n"+ returnCharMatrix(state))
            action = bot.step(generated)
            log("Action Selected by " + str(bot) + ": " + state.action_to_string(current_player, action))
            log("====================================================\n")
            # print("time for generate and play a move:", time.time()-start)
            if str(bot) == "custom":
                save_to_csv("./temp/games/"+str(bots[0])+"-"+str(bots[1])+str(game_num)+".csv", data_for_games(move, state, generated, bot.evaluator))
            
        else:
            action = bot.step(state)
        action_str = state.action_to_string(current_player, action)
        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
            if str(bot) == "custom" or str(bot) == "hunter" or str(bot) == "mcts":
                bot.update_knowledge(state.clone(), action)

        history.append(action_str)
        allStates.append(stateIntoCharMatrix(state))
        move+=1
        state.apply_action(action)

    pieces0 = 0
    pieces1 = 0
    players_piece = players_pieces()
    for i in str(state)[:100].upper():
        if i in players_piece[0]:
            pieces0 += 1
        if i in players_piece[1]:
            pieces1 += 1
    # Game is now done
    returns = state.returns()
    print("\nReturns:", 
        " ".join(map(str, returns)), "\n# moves:", len(history))
    for bot in bots:
        if str(bot) == "custom":
            hash = bot.evaluator.hash_count
            nbr_of_prior = hash[2]
            per_of_hit = round((hash[3]/hash[2])*100, 3)
            print("Hit percentage of hash in prior: ", per_of_hit, nbr_of_prior)
    for bot in bots:
        bot.restart()
    return returns, history, allStates, move, nbr_of_prior, per_of_hit, [pieces0, pieces1]

def play_n_games(player1, player2, num_games, replay=False, auto=False):
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
            returns, history, allStates, move, nbr_of_prior, per_of_hit, pieces = _play_game(game, bots, game_num)
            time_taken = round(time.time()-start_game_time)
            print("Time for this game in seconde:", time_taken)
            print("Game Number:", game_num)

            saveGame("games/"+player1+"-"+player2+str(game_num), allStates)
            # In case of tie, both win_1 and win_2 are equal to 0, otherwise the winner is 1, the other is 0
            data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num], 'time_taken': [time_taken], 'moves': [move], 'nbr_of_prior': [nbr_of_prior], 'per_of_hit': [per_of_hit], 'win_1': 1 if returns[0]==1.0 else 0, 'win_2': 1 if returns[1]==1.0 else 0, 'pieces_1': [pieces[0]], 'pieces_2': [pieces[1]]}
            save_to_csv("./temp/games/stats.csv", data)
            
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
    print("")
    print("Number of games played:", game_num + 1)
    avg_move = moves//(game_num+1)
    print("Average number of moves till finish:", avg_move)
    avg_time_taken = (round(time.time()-start_time))//(game_num+1)
    print("Average time till finish:", avg_time_taken)
    print("Players:", player1, player2)
    print("Overall wins", overall_wins)
    data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num+1], 'time_taken': [avg_time_taken], 'moves': [avg_move], 'nbr_of_prior': [0], 'per_of_hit': [0], 'win_1': [int((overall_wins[0]/(game_num+1))*100)], 'win_2': [int((overall_wins[1]/(game_num+1))*100)], 'pieces_1': [0], 'pieces_2': [0]}
    save_to_csv("./temp/games/stats.csv", data)
    return data

def benchmark(num_games):
    """With the num_games with 50, effectively each bot will play 100 games versus other bots"""
    bots_to_play = ["custom", "asmodeus", "hunter", "rnad", "mcts"]
    for i in range(4, len(bots_to_play)):
        for j in range(0, len(bots_to_play)):
            if i != j:
                player1 = bots_to_play[i]
                player2 = bots_to_play[j]
                print(player1 + " VS " + player2 + ": Begin")
                play_n_games(player1, player2, num_games)
    print("Benchmark finished its execution, congratulation !")

def evaluate_bot(bot, num_games):
    """Evaluate the bot against all the other bots"""
    # bots_to_play = ["custom", "asmodeus", "hunter", "rnad", "mcts", "basic"]
    bots_to_play = ["asmodeus", "hunter", "basic"]
    win = 0
    for i in range(len(bots_to_play)):
        player2 = bots_to_play[i]
        player1 = bot
        print(player1 + " VS " + player2 + ": Begin")
        data = play_n_games(player1, player2, num_games)
        print("Winrate VS " + player2 + ":", data["win_1"][0], "%")
        win += data["win_1"][0]
    win = int(win/len(bots_to_play))
    print("Total winrate of the bot:", win)
    print("Evaluation finished its execution, congratulation !")
    print("===========================================================")

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(None)  # Seed
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "rnad":
        try:
            state = "state.pkl"
            bot = rnadBot.rnadBot().getSavedState("states/"+state)
            print("Bot rnad loaded")
            return bot
        except:
            print("Bot rnad " + state + " failed to load so load the base rnad bot")
            return rnadBot.rnadBot()
    if bot_type == "basic":
        return basicAIBot.basicAIBot(player_id)
    if bot_type == "asmodeus":
        return asmodeusBot.asmodeusBot(player_id)
    if bot_type == "hunter":
        return hunterBot.hunterBot(player_id)
    if bot_type == "gpt":
        return gptBot.gptBot(player_id)
    if bot_type == "doi":
        return human.HumanBot()
    if bot_type == "custom":
        # uct_c parameter:
        # 0.5 give the same weight to the two part
        # 0.1 give more weight to the win
        # 0.9 give more weight to the number of time visited
        return customBot.CustomBot(game, 0.2, 50, customBot.CustomEvaluator(9, 5), player_id) # Change line 220 too
    if bot_type == "mcts":
        # uct, n_simu, bot(n_rollout, n_moves_before)
        return mctsBot.mctsBot(game, 0.5, 100, mctsBot.RandomRolloutEvaluator(75, 10), player_id)
    raise ValueError("Invalid bot type: %s" % bot_type)

if __name__ == "__main__":
    # evaluate_bot("custom", 10)
    # benchmark(50)

    # try:
    #     open("temp/log.txt", 'w').close() # Del log file
    # except IOError:
    #     print('Failure')
    ### Launch n games, params: player1, player2, game_nums, replay, auto
    # play_n_games("custom", "asmodeus", 1, replay=False, auto=False)
    # play_n_games("custom", "hunter", 1, replay=False, auto=False)
    # play_n_games("asmodeus", "hunter", 5, replay=False, auto=False)
    # script_md_evaluate_bot("games/")

    ###### Watch a game played:
    # player1 = "custom"
    # player2 = "asmodeus" # hunter asmodeus
    # game_num = 8
    # folder = "games/11-0.2/"
    # wrapper(print_board, getGame(folder+player1+"-"+player2+str(game_num)), [player1, player2], auto=False)
    
    ###### Train the Rnad a number of steps
    # everythingEverywhereAllAtOnce("states/state.pkl", 10000) # 100000, 3sec/step

    # play_game_versus_doi(0, "doi", "custom")
    # /bin/python3 /home/thomas/Bureau/tfe/main.py < temp/t.txt
    # Bureau/stratego-0.13.4/src$ ./run_stratego.sh 

    # script_md_evaluate_benchmark("bench-final-nodoi/")
    script_md_evaluate_benchmark("games/")
