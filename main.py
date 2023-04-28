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
import matplotlib.pyplot as plt
import markdown
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

def script_md_evaluate_bot(folder):
    # global_stats add 
    stats_df = pd.read_csv(folder+"stats.csv")
    markdown_content = []

    # Create a dictionary to store the game data
    global_stats = {}
    eval_matrix_win = None
    eval_matrix_lose = None
    unk_acc_win = None
    unk_acc_lose = None

    # Iterate over the rows of the stats dataframe and read in the corresponding game csv files
    for index, row in stats_df.iterrows():
        player1 = row['player1']
        player2 = row['player2']
        game_num = row['game_num']
        win_1 = row['win_1']
        win_2 = row['win_2']
        
        if win_1 < 2:
            # Create the filename for the game csv file
            filename = f"{player1}-{player2}{game_num}.csv"
            # Read in the game csv file and store it in the games dictionary
            game_df = pd.read_csv(folder+filename)
            eval_real = np.array(game_df["eval_real"])
            padded_eval_real = np.pad(eval_real, (0, 1001-len(eval_real)), mode='constant', constant_values=np.nan)
            unk_acc = np.array(game_df["unknow_acc"])
            padded_unk_acc = np.pad(unk_acc, (0, 1001-len(unk_acc)), mode='constant', constant_values=np.nan)

            if win_1:
                if eval_matrix_win is None:
                    eval_matrix_win = padded_eval_real
                    unk_acc_win = padded_unk_acc
                else:
                    eval_matrix_win = np.vstack((eval_matrix_win, padded_eval_real))
                    unk_acc_win = np.vstack((unk_acc_win, padded_unk_acc))
            elif win_2:
                if eval_matrix_lose is None:
                    eval_matrix_lose = padded_eval_real
                    unk_acc_lose = padded_unk_acc
                else:
                    eval_matrix_lose = np.vstack((eval_matrix_lose, padded_eval_real))
                    unk_acc_lose = np.vstack((unk_acc_lose, padded_unk_acc))
            # Draw not counted, turn the elif above into a if to take them in the lose one
        else:
            avg_time = row['time_taken']
            avg_moves = row['moves']
            # Add the stats versus this bot to the global stats
                                            # Nombre de game, temps moyen, moves moyen, win%, draw%, lose%
            global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])

    for matrix, acc, win in [(eval_matrix_win, unk_acc_win, "win"), (eval_matrix_lose, unk_acc_lose, "lose")]:
        x_values = range(0, 2001, 2)
        # Compute the mean and standard deviation for each index
        mean_values = [np.nanmean(matrix, axis=0), np.nanmean(acc, axis=0)]
        std_values = [np.nanstd(matrix, axis=0), np.nanstd(acc, axis=0)]

        # Plot the graph
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(x_values, mean_values[0])
        axs[0].fill_between(x_values, mean_values[0] - std_values[0], mean_values[0] + std_values[0], alpha=0.3)
        axs[0].set_xlabel('Moves')
        axs[0].set_ylabel('Evaluation')
        axs[0].set_title('Evaluation of the state over the moves: '+win)

        axs[1].plot(x_values, mean_values[1])
        axs[1].fill_between(x_values, mean_values[1] - std_values[1], mean_values[1] + std_values[1], alpha=0.3)
        axs[1].set_xlabel('Moves')
        axs[1].set_ylabel('Unknown accuracy')
        axs[1].set_title('Accuracy of unknown pieces over the moves: '+win)
        
        # plt.show()
        fig.savefig(folder+f"eval-{win}.png")

    markdown_content.append(f"# Changement: {folder}\n")

    markdown_content.append("## Evaluation of the state over the moves: win\n")
    markdown_content.append(f"\n![{folder}eval-win.png Graph](./eval-win.png)\n")

    markdown_content.append("## Evaluation of the state over the moves: lose\n")
    markdown_content.append(f"\n![{folder}eval-lose.png Graph](./eval-lose.png)\n")

    markdown_content.append("## Win/Draw/Lose versus the bots\n")
    markdown_content.append(f"\n| ... | {player1} |\n")
    markdown_content.append(f"| --- | --- |\n")
    for e_player, information in global_stats.items():
        # global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])
        game_num, avg_time, avg_moves, win, draw, lose = information
        markdown_content.append(f"| {e_player} | {win}/{draw}/{lose} |\n")

    markdown_content.append("## Other miscellanous stats versus the bots\n")
    markdown_content.append(f"\n| {player1} | Nbr Games | Avg_time | Avg_moves |\n")
    markdown_content.append(f"| --- | --- | --- | --- |\n")
    for e_player, information in global_stats.items():
        # global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])
        game_num, avg_time, avg_moves, win, draw, lose = information
        markdown_content.append(f"| {e_player} | {game_num} | {avg_time} | {avg_moves} |\n")

    # Write the markdown content to a file
    with open(folder+'results.md', 'w') as f:
        f.write(''.join(markdown_content))

def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(None)  # Seed
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "custom":
        return customBot.CustomBot(game, 1.5, 75, customBot.CustomEvaluator(), player_id)
    if bot_type == "mcts":
        return mctsBot.mctsBot(game, 1.5, 75, mctsBot.RandomRolloutEvaluator(), player_id)
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
    raise ValueError("Invalid bot type: %s" % bot_type)

def _play_game(game, bots, game_num):
    """Plays one game."""
    # state = game.new_initial_state("FEBMBEFEEFBGIBHIBEDBGJDDDHCGJGDHDLIFKDDHAA__AA__AAAA__AA__AATPPWRUXPTPSVSOTPPPVSNPQNUTNUSNRQQRQNYNQR r 0") # Equal state
    state = game.new_initial_state(create_initial_state_str())
    history = []
    allStates = []

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
                bot.restart()
            return returns, history, allStates, move, [pieces0, pieces1]            

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
            save_to_csv("./games/"+str(bots[0])+"-"+str(bots[1])+str(game_num)+".csv", data_for_games(move, state, generated, customBot.CustomEvaluator()))
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
        bot.restart()
    return returns, history, allStates, move, [pieces0, pieces1]

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
            returns, history, allStates, move, pieces = _play_game(game, bots, game_num)
            time_taken = round(time.time()-start_game_time)
            print("Time for this game in seconde:", time_taken)
            print("Game Number:", game_num)

            saveGame("games/"+player1+"-"+player2+str(game_num), allStates)
            # In case of tie, both win_1 and win_2 are equal to 0, otherwise the winner is 1, the other is 0
            data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num], 'time_taken': [time_taken], 'moves': [move], 'win_1': 1 if returns[0]==1.0 else 0, 'win_2': 1 if returns[1]==1.0 else 0, 'pieces_1': [pieces[0]], 'pieces_2': [pieces[1]]}
            save_to_csv("./games/stats.csv", data)
            
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
    data = {'player1': [player1], 'player2': [player2], 'game_num': [game_num+1], 'time_taken': [avg_time_taken], 'moves': [avg_move], 'win_1': [int((overall_wins[0]/(game_num+1))*100)], 'win_2': [int((overall_wins[1]/(game_num+1))*100)], 'pieces_1': [0], 'pieces_2': [0]}
    save_to_csv("./games/stats.csv", data)
    return data

def benchmark(num_games):
    """With the num_games with 50, effectively each bot will play 100 games versus other bots"""
    # bots_to_play = ["custom", "asmodeus", "hunter", "rnad", "mcts"]
    bots_to_play = ["basic", "rnad"]
    for i in range(len(bots_to_play)):
        for j in range(len(bots_to_play)):
            if i != j:
                player1 = bots_to_play[i]
                player2 = bots_to_play[j]
                print(player1 + " VS " + player2 + ": Begin")
                play_n_games(player1, player2, num_games)
                print(player1 + " VS " + player2 + ": Finished")
    print("Benchmark finished its execution, congratulation !")

def evaluate_bot(bot, num_games):
    """Evaluate the bot against all the other bots"""
    # bots_to_play = ["custom", "asmodeus", "hunter", "rnad", "mcts"]
    bots_to_play = ["hunter", "asmodeus", "basic", "rnad"]
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
    

if __name__ == "__main__":
    ###### Launch only n games, params: player1, player2, game_nums, replay, auto
    # play_n_games("custom", "asmodeus", 1, replay=False, auto=False)

    # evaluate_bot("custom", 5)
    script_md_evaluate_bot("games/prot/")
    # play_n_games("custom", "basic", 20, replay=False, auto=False)

    # benchmark(5)

    ###### Watch a game played
    # player1 = "custom"
    # player2 = "hunter"
    # game_num = 4
    # wrapper(print_board, getGame("games/"+player1+"-"+player2+str(game_num)), [player1, player2], auto=False)
    # wrapper(print_board, getGame("games/normal/"+player1+"-"+player2+str(game_num)), [player1, player2], auto=False)
    # wrapper(print_board, getGame("games/simenemyprior/"+player1+"-"+player2+str(game_num)), [player1, player2], auto=False)
    # wrapper(print_board, getGame("games/protscaleprio/"+player1+"-"+player2+str(game_num)), [player1, player2], auto=False)

    ###### Train the Rnad a number of steps
    # everythingEverywhereAllAtOnce("states/state.pkl", 10000) # 100000, 3sec/step


# A chaque entrée de mon tableau à double entrée win/draw/lose

# Tester les heuristic avec des graphs
    # Faire des stats avec des evolution d'heuristic. Prendre des moyennes de l'evolution de l'heuristic
    # Grouper par win / lose
    # Generer des graphs a chaque changement pour voir les reels changement directement 
    # Automatiser tout ça pour voir. Faire des scripts pour voir.
    # Générer un .md avec les différents graphs et les stats de win / lose sous forme de tableau, nombre de coups, etc directement.

# Prendre l'heuristic d'un autre bot et le mettre dans notre monte-carlo pour voir l'impact de ça

# Faire le demon of ignorance comme bot en plus ou au moins avoir son heuristic et comment il fait des trucs pour comparer

# Mettre mon heuristic à moi dans un alpha-beta pour à nouveau voir l'impact du monte-carlo sur l'heuristic

# Regarder comment on choisis un move et comment les autres bots choisissent leur move. Regarder comment les autres bots evaluent leur %win



