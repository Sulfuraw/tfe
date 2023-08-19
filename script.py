import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def script_md_evaluate_bot(folder):
    stats_df = pd.read_csv(folder+"stats.csv")
    markdown_content = []

    # Create a dictionary to store the game data
    global_stats = {}
    eval_matrix_win = None
    eval_matrix_lose = None
    unk_acc_win = None
    unk_acc_lose = None
    games_stats = {}

    # Iterate over the rows of the stats dataframe and read in the corresponding game csv files
    for index, row in stats_df.iterrows():
        player1 = row['player1']
        player2 = row['player2']
        game_num = row['game_num']
        win_1 = row['win_1']
        win_2 = row['win_2']
        avg_time = row['time_taken']
        avg_moves = row['moves']
        pieces_1 = row['pieces_1']
        pieces_2 = row['pieces_2']

        if win_1 < 2 and win_2 < 2:
            filename = f"{player1}-{player2}{game_num}.csv"
            game_df = pd.read_csv(folder+filename)
            games_stats[filename] = np.array([avg_moves, win_1-win_2, pieces_1, pieces_2])

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
            # Add the stats versus this bot to the global stats
                                            # Nombre de game, temps moyen, moves moyen, win%, draw%, lose%
            global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])

    for matrix, acc, win in [(eval_matrix_win, unk_acc_win, "win"), (eval_matrix_lose, unk_acc_lose, "lose")]:
        if acc is None:
            continue
        x_values = range(0, 2001, 2)
        # Compute the mean and standard deviation for each index
        mean_values = [np.nanmean(matrix, axis=0), np.nanmean(acc, axis=0)]

        above_mean = [matrix[matrix > mean_values[0]], acc[acc > mean_values[1]]]
        below_mean = [matrix[matrix < mean_values[0]], acc[acc < mean_values[1]]]

        std_values_above = [np.nanstd(above_mean[0], axis=0), np.nanstd(above_mean[1], axis=0)]
        std_values_below = [np.nanstd(below_mean[0], axis=0), np.nanstd(below_mean[1], axis=0)]

        # Plot the graph
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(x_values, mean_values[0])
        axs[0].fill_between(x_values, mean_values[0] - std_values_below[0], mean_values[0] + std_values_above[0], alpha=0.3)
        axs[0].set_xlabel('Moves')
        axs[0].set_ylabel('Evaluation')
        axs[0].set_title('Evaluation of the state over the moves: '+win)
        axs[0].set_ylim(-1, 1)

        axs[1].plot(x_values, mean_values[1])
        axs[1].fill_between(x_values, mean_values[1] - std_values_below[1], mean_values[1] + std_values_above[1], alpha=0.3)
        axs[1].set_xlabel('Moves')
        axs[1].set_ylabel('Unknown accuracy')
        axs[1].set_title('Accuracy of unknown pieces over the moves: '+win)
        axs[1].set_ylim(0, 1)
        
        # plt.show()
        fig.savefig(folder+f"eval-{win}.png")

    markdown_content.append(f"# Changement: {folder}\n")

    markdown_content.append("### Evaluation of the state over the moves: win\n")
    markdown_content.append(f"\n![{folder}eval-win.png Graph](./eval-win.png)\n")

    markdown_content.append("### Evaluation of the state over the moves: lose\n")
    markdown_content.append(f"\n![{folder}eval-lose.png Graph](./eval-lose.png)\n")

    markdown_content.append("## Win/Draw/Lose versus the bots\n")
    markdown_content.append(f"\n| ... | {player1} |\n")
    markdown_content.append(f"| --- | --- |\n")
    for e_player, information in global_stats.items():
        game_num, avg_time, avg_moves, win, draw, lose = information
        markdown_content.append(f"| {e_player} | {win}/{draw}/{lose} |\n")

    markdown_content.append("## Other miscellanous stats versus the bots\n")
    markdown_content.append(f"\n| {player1} | Nbr Games | Avg_time | Avg_moves |\n")
    markdown_content.append(f"| --- | --- | --- | --- |\n")
    for e_player, information in global_stats.items():
        game_num, avg_time, avg_moves, win, draw, lose = information
        markdown_content.append(f"| {e_player} | {game_num} | {avg_time} | {avg_moves} |\n")

    markdown_content.append("## Stats per game\n")
    markdown_content.append(f"\n| Game | Nbr_moves | Win | pieces_1 | pieces_2 |\n")
    markdown_content.append(f"| --- | --- | --- | --- | --- |\n")
    for filename, information in games_stats.items():
        move, win, pieces_1, pieces_2 = information
        markdown_content.append(f"| {filename} | {move} | {win} | {pieces_1} | {pieces_2} |\n")

    markdown_content.append("## Post-game Observation\n")
    markdown_content.append("To fill...\n")

    # Write the markdown content to a file
    with open(folder+'results.md', 'w') as f:
        f.write(''.join(markdown_content))

##########################################################################

def get_stats(stats_df, folder, bot):
    # Create a dictionary to store the game data
    global_stats = {}
    eval_matrix_win = None
    eval_matrix_lose = None
    unk_acc_win = None
    unk_acc_lose = None
    games_stats = {}

    # Iterate over the rows of the stats dataframe and read in the corresponding game csv files
    for index, row in stats_df.iterrows():
        player1 = row['player1']
        player2 = row['player2']

        if player1 != bot and player2 != bot:
            continue

        game_num = row['game_num']
        win_1 = row['win_1']
        win_2 = row['win_2']
        avg_time = row['time_taken']
        avg_moves = row['moves']
        pieces_1 = row['pieces_1']
        pieces_2 = row['pieces_2']
        if win_1 < 2 and win_2 < 2:
            games_stats[f"{player1}-{player2}{game_num}"] = np.array([avg_moves, win_1-win_2, pieces_1, pieces_2])

        if win_1 < 2 and win_2 < 2 and (player1 == "custom" or player2 == "custom"):
            filename = f"{player1}-{player2}{game_num}.csv"
            game_df = pd.read_csv(folder+filename)

            eval_real = np.array(game_df["eval_real"])
            padded_eval_real = np.pad(eval_real, (0, 751-len(eval_real)), mode='constant', constant_values=np.nan)
            unk_acc = np.array(game_df["unknow_acc"])
            padded_unk_acc = np.pad(unk_acc, (0, 751-len(unk_acc)), mode='constant', constant_values=np.nan)

            if (win_1 and player1 == "custom") or (win_2 and player2 == "custom"):
                if eval_matrix_win is None:
                    eval_matrix_win = padded_eval_real
                    unk_acc_win = padded_unk_acc
                else:
                    eval_matrix_win = np.vstack((eval_matrix_win, padded_eval_real))
                    unk_acc_win = np.vstack((unk_acc_win, padded_unk_acc))
            # Draw not counted, turn the elif above into a if to take them in the lose one
            elif (win_1 and player2 == "custom") or (win_2 and player1 == "custom"):
                if eval_matrix_lose is None:
                    eval_matrix_lose = padded_eval_real
                    unk_acc_lose = padded_unk_acc
                else:
                    eval_matrix_lose = np.vstack((eval_matrix_lose, padded_eval_real))
                    unk_acc_lose = np.vstack((unk_acc_lose, padded_unk_acc))
        elif win_1 > 2 or win_2 > 2:
            # Add the stats versus this bot to the global stats
            if player1 == bot:
                if global_stats.get(player2) is None:
                    global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])
                else:
                    global_stats[player2] = (global_stats[player2] + np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2]))/2
            else:
                if global_stats.get(player1) is None:
                    global_stats[player1] = np.array([game_num, avg_time, avg_moves, win_2, 100-win_1-win_2, win_1])
                else:
                    global_stats[player1] = (global_stats[player1] + np.array([game_num, avg_time, avg_moves, win_2, 100-win_1-win_2, win_1]))/2
        for opponent, stats in global_stats.items():
            global_stats[opponent] = np.around(global_stats[opponent], 0)
            global_stats[opponent] = np.asarray(global_stats[opponent], dtype = 'int')
        if eval_matrix_win is not None:
            eval_matrix_win = np.around(eval_matrix_win, 3)
        if eval_matrix_lose is not None:
            eval_matrix_lose = np.around(eval_matrix_lose, 3)
        if unk_acc_win is not None:
            unk_acc_win = np.around(unk_acc_win, 2)
        if unk_acc_lose is not None:
            unk_acc_lose = np.around(unk_acc_lose, 2)
    return global_stats, eval_matrix_win, eval_matrix_lose, unk_acc_win, unk_acc_lose, games_stats


def get_stats_per_bot(stats_df, folder, bot, bot2):
    # Create a dictionary to store the game data
    global_stats = {}
    eval_matrix_win = None
    eval_matrix_lose = None
    unk_acc_win = None
    unk_acc_lose = None
    games_stats = {}

    # Iterate over the rows of the stats dataframe and read in the corresponding game csv files
    for index, row in stats_df.iterrows():
        player1 = row['player1']
        player2 = row['player2']

        if player1 != bot and player2 != bot:
            continue
        if player1 != bot2 and player2 != bot2:
            continue

        game_num = row['game_num']
        win_1 = row['win_1']
        win_2 = row['win_2']
        avg_time = row['time_taken']
        avg_moves = row['moves']
        pieces_1 = row['pieces_1']
        pieces_2 = row['pieces_2']
        games_stats[f"{player1}-{player2}{game_num}"] = np.array([avg_moves, win_1-win_2, pieces_1, pieces_2])

        if win_1 < 2 and win_2 < 2 and (player1 == "custom" or player2 == "custom"):
            filename = f"{player1}-{player2}{game_num}.csv"
            game_df = pd.read_csv(folder+filename)

            eval_real = np.array(game_df["eval_real"])
            padded_eval_real = np.pad(eval_real, (0, 751-len(eval_real)), mode='constant', constant_values=np.nan)
            unk_acc = np.array(game_df["unknow_acc"])
            padded_unk_acc = np.pad(unk_acc, (0, 751-len(unk_acc)), mode='constant', constant_values=np.nan)

            if (win_1 and player1 == "custom") or (win_2 and player2 == "custom"):
                if eval_matrix_win is None:
                    eval_matrix_win = padded_eval_real
                    unk_acc_win = padded_unk_acc
                else:
                    eval_matrix_win = np.vstack((eval_matrix_win, padded_eval_real))
                    unk_acc_win = np.vstack((unk_acc_win, padded_unk_acc))
            # Draw not counted, turn the elif above into a if to take them in the lose one
            elif (win_1 and player2 == "custom") or (win_2 and player1 == "custom"):
                if eval_matrix_lose is None:
                    eval_matrix_lose = padded_eval_real
                    unk_acc_lose = padded_unk_acc
                else:
                    eval_matrix_lose = np.vstack((eval_matrix_lose, padded_eval_real))
                    unk_acc_lose = np.vstack((unk_acc_lose, padded_unk_acc))
        else:
            # Add the stats versus this bot to the global stats
            if player1 == bot:
                if global_stats.get(player2) is None:
                    global_stats[player2] = np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2])
                else:
                    global_stats[player2] = (global_stats[player2] + np.array([game_num, avg_time, avg_moves, win_1, 100-win_1-win_2, win_2]))/2
            else:
                if global_stats.get(player1) is None:
                    global_stats[player1] = np.array([game_num, avg_time, avg_moves, win_2, 100-win_1-win_2, win_1])
                else:
                    global_stats[player1] = (global_stats[player1] + np.array([game_num, avg_time, avg_moves, win_2, 100-win_1-win_2, win_1]))/2
        for opponent, stats in global_stats.items():
            global_stats[opponent] = np.around(global_stats[opponent], 0)
            global_stats[opponent] = np.asarray(global_stats[opponent], dtype = 'int')
        if eval_matrix_win is not None:
            eval_matrix_win = np.around(eval_matrix_win, 3)
        if eval_matrix_lose is not None:
            eval_matrix_lose = np.around(eval_matrix_lose, 3)
        if unk_acc_win is not None:
            unk_acc_win = np.around(unk_acc_win, 2)
        if unk_acc_lose is not None:
            unk_acc_lose = np.around(unk_acc_lose, 2)
    return global_stats, eval_matrix_win, eval_matrix_lose, unk_acc_win, unk_acc_lose, games_stats

def script_md_evaluate_benchmark(folder):
    stats_df = pd.read_csv(folder+"stats.csv")
    markdown_content = []
    markdown_content.append(f"# Changement: {folder}\n")
    whole_global_stats = {}
    whole_games_stats = {}

    for bot in ["custom", "asmodeus", "hunter", "rnad", "mcts", "doi"]:
        global_stats, eval_matrix_win, eval_matrix_lose, unk_acc_win, unk_acc_lose, games_stats = get_stats(stats_df, folder, bot)
        whole_global_stats[bot] = global_stats
        whole_games_stats.update(games_stats)
        if bot == "custom":
            for matrix, acc, win in [(eval_matrix_win, unk_acc_win, "win"), (eval_matrix_lose, unk_acc_lose, "lose")]:
                if acc is None:
                    continue
                x_values = range(0, 1501, 2)
                # Compute the mean and standard deviation for each index
                mean_values = [np.nanmean(matrix, axis=0), np.nanmean(acc, axis=0)]

                above_mean = [matrix[matrix > mean_values[0]], acc[acc > mean_values[1]]]
                below_mean = [matrix[matrix < mean_values[0]], acc[acc < mean_values[1]]]

                std_values_above = [np.nanstd(above_mean[0], axis=0), np.nanstd(above_mean[1], axis=0)]
                std_values_below = [np.nanstd(below_mean[0], axis=0), np.nanstd(below_mean[1], axis=0)]

                # Plot the graph
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].plot(x_values, mean_values[0])
                axs[0].fill_between(x_values, mean_values[0] - std_values_below[0], mean_values[0] + std_values_above[0], alpha=0.3)
                axs[0].set_xlabel('Moves')
                axs[0].set_ylabel('Evaluation')
                axs[0].set_title('Evaluation of the state over the moves: '+win)
                axs[0].set_ylim(-1, 1)

                axs[1].plot(x_values, mean_values[1])
                axs[1].fill_between(x_values, mean_values[1] - std_values_below[1], mean_values[1] + std_values_above[1], alpha=0.3)
                axs[1].set_xlabel('Moves')
                axs[1].set_ylabel('Unknown accuracy')
                axs[1].set_title('Accuracy of unknown pieces over the moves: '+win)
                axs[1].set_ylim(0, 1)
                
                # plt.show()
                fig.savefig(folder+f"eval-{bot}-{win}.png")
            for bot2 in ["asmodeus", "hunter", "rnad", "mcts", "doi"]:
                global_stats, eval_matrix_win, eval_matrix_lose, unk_acc_win, unk_acc_lose, games_stats = get_stats_per_bot(stats_df, folder, bot, bot2)
                for matrix, acc, win in [(eval_matrix_win, unk_acc_win, "win"), (eval_matrix_lose, unk_acc_lose, "lose")]:
                    if acc is None:
                        continue
                    x_values = range(0, 1501, 2)
                    # Compute the mean and standard deviation for each index
                    mean_values = [np.nanmean(matrix, axis=0), np.nanmean(acc, axis=0)]

                    above_mean = [matrix[matrix > mean_values[0]], acc[acc > mean_values[1]]]
                    below_mean = [matrix[matrix < mean_values[0]], acc[acc < mean_values[1]]]

                    std_values_above = [np.nanstd(above_mean[0], axis=0), np.nanstd(above_mean[1], axis=0)]
                    std_values_below = [np.nanstd(below_mean[0], axis=0), np.nanstd(below_mean[1], axis=0)]

                    # Plot the graph
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].plot(x_values, mean_values[0])
                    axs[0].fill_between(x_values, mean_values[0] - std_values_below[0], mean_values[0] + std_values_above[0], alpha=0.3)
                    axs[0].set_xlabel('Moves')
                    axs[0].set_ylabel('Evaluation')
                    axs[0].set_title('Evaluation of the state over the moves: '+win)
                    axs[0].set_ylim(-1, 1)

                    axs[1].plot(x_values, mean_values[1])
                    axs[1].fill_between(x_values, mean_values[1] - std_values_below[1], mean_values[1] + std_values_above[1], alpha=0.3)
                    axs[1].set_xlabel('Moves')
                    axs[1].set_ylabel('Unknown accuracy')
                    axs[1].set_title('Accuracy of unknown pieces over the moves: '+win)
                    axs[1].set_ylim(0, 1)
                    
                    # plt.show()
                    fig.savefig(folder+f"eval-{bot}-{bot2}-{win}.png")

            winrate = [0, 0, 0]
            nbr_games = 0
            for e_player, information in whole_global_stats["custom"].items():
                game_num, avg_time, avg_moves, win, draw, lose = information
                winrate[0] += win*game_num
                winrate[1] += draw*game_num
                winrate[2] += lose*game_num
                nbr_games += game_num
            winrate = np.array(winrate)/nbr_games

            markdown_content.append(f"## Global\n")
            markdown_content.append(f"Total winrate of custom: {winrate[0]} / {winrate[1]} / {winrate[2]}\n")
            markdown_content.append("### Evaluation of the state over the moves: win\n")
            markdown_content.append(f"\n![{folder}eval-{bot}-win.png Graph](./eval-{bot}-win.png)\n")

            markdown_content.append("### Evaluation of the state over the moves: lose\n")
            markdown_content.append(f"\n![{folder}eval-{bot}-lose.png Graph](./eval-{bot}-lose.png)\n")

        markdown_content.append(f"## {bot}\n")
        if bot != "custom":
            markdown_content.append("### Evaluation of the state over the moves: win\n")
            markdown_content.append(f"\n![{folder}eval-custom-{bot}-win.png Graph](./eval-custom-{bot}-win.png)\n")

            markdown_content.append("### Evaluation of the state over the moves: lose\n")
            markdown_content.append(f"\n![{folder}eval-custom-{bot}-lose.png Graph](./eval-custom-{bot}-lose.png)\n")
        
        markdown_content.append(f"### Win/Draw/Lose of {bot}\n")
        markdown_content.append(f"\n| ... | {bot} |\n")
        markdown_content.append(f"| --- | --- |\n")
        for e_player, information in whole_global_stats[bot].items():
            game_num, avg_time, avg_moves, win, draw, lose = information
            markdown_content.append(f"| {e_player} | {win}/{draw}/{lose} |\n")

        markdown_content.append("### Other miscellanous stats\n")
        markdown_content.append(f"\n| {bot} | Nbr Games | Avg_time | Avg_moves |\n")
        markdown_content.append(f"| --- | --- | --- | --- |\n")
        for e_player, information in whole_global_stats[bot].items():
            game_num, avg_time, avg_moves, win, draw, lose = information
            markdown_content.append(f"| {e_player} | {game_num*2} | {avg_time} | {avg_moves} |\n")

    markdown_content.append("## Stats per game\n")
    markdown_content.append(f"\n| Game | Nbr_moves | Win | pieces_1 | pieces_2 |\n")
    markdown_content.append(f"| --- | --- | --- | --- | --- |\n")
    for filename, information in whole_games_stats.items():
        move, win, pieces_1, pieces_2 = information
        markdown_content.append(f"| {filename} | {move} | {win} | {pieces_1} | {pieces_2} |\n")

    markdown_content.append("## Post-game Observation\n")
    markdown_content.append("To fill...\n")

    # Write the markdown content to a file
    with open(folder+'results.md', 'w') as f:
        f.write(''.join(markdown_content))