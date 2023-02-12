import curses
import numpy as np

players_piece = [["M", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
                 ["Y", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]]
# It is:         [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
# Nbr of each:   [1,   6,    1,   8,   5,   4,   4,   4,   3,   2,   1,   1]

# Proba if we have no info
def no_info_piece(nbr_piece_left):
    return nbr_piece_left/sum(nbr_piece_left)

# Proba if we know this piece moved before
def moved_piece(nbr_piece_left):
    proba = np.array([0.0]*12)
    proba[2:] = nbr_piece_left[2:]/sum(nbr_piece_left[2:])
    return proba

def action_to_coord(action_str):
    """ Returns coord in the form: [col1, row1, col2, row2]
        We will use it in the form: matrix[row1][col1] to matrix[row2][col2]
        i.e:
            coord = action_to_coord(action_str)
            matrix[coord[1]][coord[0]] --->  matrix[coord[3]][coord[2]]
    """
    rocols = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "k":9, "1":0, "2":1, "3":2, "4":3, "5":4, "6":5, "7":6, "8":7, "9":8, ":":9}
    pos = list(action_str)
    returns = []
    for i in range(len(pos)):
        returns.append(rocols[pos[i]])
    return returns

# Verify that the state_str is a valid state, to be used for our generate_state
def is_valid_state(state_str):
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
        and state_str.count("?") == 0)

# Generate a valid state, with the knowledge of our bot + the partial state
def generate_state(state, matrix_of_possibilities, information):
    partial_state = state.information_state_string(state.current_player())
    state_str = str(partial_state)
    piece_left = information[1].copy()
    moved_before = information[2]
    
    final = ""

    # print("==============================================")
    # printCharMatrix(stateIntoCharMatrix(str(state)))
    # print()
    # printCharMatrix(stateIntoCharMatrix(state_str))
    # print()
    # print(matrix_of_possibilities[5:])
    # print(information[1])
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

# Generate the matrix of possibilites / probabilities used for piece generation in the generate_state
def generate_possibilities_matrix(state, information):
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

# Each move a player does can affect the information / knowledge that our bot has
def updating_knowledge(information, state, action):
    player_id, nbr_piece_left, moved_before, moved_scout = information
    current_player = state.current_player()

    coord = action_to_coord(state.action_to_string(current_player, action))

    full_matrix_before = stateIntoCharMatrix(state)
    matrix_before = stateIntoCharMatrix(state.information_state_string(player_id))
    start = matrix_before[coord[1]][coord[0]]
    arrival = matrix_before[coord[3]][coord[2]]

    state.apply_action(action)
    matrix_after = stateIntoCharMatrix(state.information_state_string(player_id))
    arrival_after = matrix_after[coord[3]][coord[2]]

    if current_player == player_id:
        # Fight
        # if arrival in players_piece[1-player_id]: # Isn't needed because if that's the case, we know it and deleted info
        # About it already
        if arrival == "?":
            # Either win or lose the fight, it will result in the same operation
            # We -1 the count and generate_possibilities_matrix will do the work thanks to information_state_string
            # We also delete information of moved piece at this place because it doesnt matter anymore
            for p in range(12):
                if full_matrix_before[coord[3]][coord[2]] == players_piece[1-player_id][p]:
                    nbr_piece_left[p] -= 1
                    moved_before[coord[3]][coord[2]] = 0
                    moved_scout[coord[3]][coord[2]] = 0
                    return np.array([player_id, nbr_piece_left, moved_before, moved_scout]) 
    else:
        # Fight
        if arrival in players_piece[player_id]:
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 0
            moved_scout[coord[1]][coord[0]] = 0
            moved_scout[coord[3]][coord[2]] = 0
            # Ennemy Won: But if he killed our piece, we get information of it
            if arrival_after in players_piece[current_player]:
                if start == "?":
                    for p in range(12):
                        if arrival_after == players_piece[current_player][p]:
                            nbr_piece_left[p] -= 1
            # Ennemy lose:
            else:
                for p in range(12):
                    if full_matrix_before[coord[1]][coord[0]] == players_piece[current_player][p] and start == "?":
                        nbr_piece_left[p] -= 1
            return np.array([player_id, nbr_piece_left, moved_before, moved_scout])
        # Deplacement on empty space: Deplacement of moved and/or get information about moved/scout_moved
        else:
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 1
            if abs(coord[1]-coord[3]) > 1 or abs(coord[0]-coord[2]) > 1:
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1
            if moved_scout[coord[1]][coord[0]]:
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1
            return np.array([player_id, nbr_piece_left, moved_before, moved_scout])
    return information

# Transform a state into a Matrix of Character inside the str(state)
def stateIntoCharMatrix(state):
    stat = str(state).upper().split(" ")[0]
    stat = ' '.join(stat)
    statInLines = []
    for i in range(0, len(stat), 20):
        statInLines.append(stat[i:i+20].rstrip().split(" "))
    return statInLines

# Tools for flag_protec
# TODO: Add the coord of Walls/Rivers in the middle as invalid
def is_valid_coord(pos):
    return (0 <= pos[0] < 10) and (0 <= pos[1] < 10)

# Check that the flag is protected all around him (4 positions)
def flag_protec(state, player):
    matrix = stateIntoCharMatrix(state)
    coord = [0, 0]
    for i in range(10):
        for j in range(10):
            if matrix[i][j] == players_piece[player][0]:
                coord = [i, j]
    protected = True
    to_check = [(coord[0]+1, coord[1]), (coord[0]-1, coord[1]), (coord[0], coord[1]+1), (coord[0], coord[1]-1)]
    for pos in to_check:
        if is_valid_coord(pos):
            protected = protected and (matrix[pos[0]][pos[1]] in players_piece[player])
    return protected 
    
# Print device for the matrix of character generated here: stateIntoCharMatrix(state)
def printCharMatrix(charMatrix):
    final = ""
    for line in charMatrix:
        for char in line:
            if char == "_":
                final += "#" + " "
            elif char == "A":
                final += "_" + " "
            else:
                final += char + " "
        final += "\n"
    print(final)

# wrapper(print_board, getGame("games/0"), ["rnad", "random"])
# To print only one state simply put it inside [] as states arg
def print_board(stdscr, states, players, auto):
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    color = curses.color_pair(1)
    stdscr.addstr(1, len(states[0][0])*2+len(states[0][0])+6, 
                    "Player 1 plays as "+str(players[0]), color)
    color = curses.color_pair(4)
    stdscr.addstr(7, len(states[0][1])*2+len(states[0][1])+6, 
                    "Player 2 plays as "+str(players[1]), color)

    color = curses.color_pair(7)
    stdscr.addstr(0, 65, "Fl = Flag (1)", color)
    stdscr.addstr(1, 65, "Bo = Bombs (6)", color)
    stdscr.addstr(2, 65, "Sp = Spy (1)", color)
    stdscr.addstr(3, 65, "Sc = Scout (8)", color)
    stdscr.addstr(4, 65, "Mi = Miners (5)", color)
    stdscr.addstr(5, 65, "Sg = Sergeant (4)", color)
    stdscr.addstr(6, 65, "Lt = Lieutenant (4)", color)
    stdscr.addstr(7, 65, "Cp = Captain (4)", color)
    stdscr.addstr(8, 65, "Mj = Major (3)", color)
    stdscr.addstr(9, 65, "Co = Colonel (2)", color)
    stdscr.addstr(10, 65, "Ge = General (1)", color)
    stdscr.addstr(11, 65, "Ms = Marshal (1)", color)

    nthState = 0
    while nthState < len(states):
        state = states[nthState]
        for i in range(len(state)):
            for j in range(len(state[i])):
                oldPiece = state[i][j].upper()
                if oldPiece in ["M", "B", "C", "D", "E", "F", 
                                "G", "H", "I", "J", "K", "L"]:
                    color = curses.color_pair(1)
                elif oldPiece in ["Y", "N", "O", "P", "Q", "R", 
                                "S", "T", "U", "V", "W", "X"]:
                    color = curses.color_pair(4)
                else: color = curses.color_pair(7)
                if oldPiece in ["M", "Y"]: piece = "Fl"
                elif oldPiece in ["B", "N"]: piece = "Bo"
                elif oldPiece in ["C", "O"]: piece = "Sp"
                elif oldPiece in ["D", "P"]: piece = "Sc"
                elif oldPiece in ["E", "Q"]: piece = "Mi"
                elif oldPiece in ["F", "R"]: piece = "Sg"
                elif oldPiece in ["G", "S"]: piece = "Lt"
                elif oldPiece in ["H", "T"]: piece = "Cp"
                elif oldPiece in ["I", "U"]: piece = "Mj"
                elif oldPiece in ["J", "V"]: piece = "Co"
                elif oldPiece in ["K", "W"]: piece = "Ge"
                elif oldPiece in ["L", "X"]: piece = "Ms"
                elif oldPiece in ["_"]: piece = "##"
                elif oldPiece in ["A"]: piece = "__"
                else: piece = "??"
                stdscr.addstr(i+1, j*2+j+2, piece, color)
        color = curses.color_pair(7)
        stdscr.addstr(i+1, j*2+j+9, "Turn number " + str(nthState), color)
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
                    nthState += 100
                    break
                if c == ord('g'):
                    nthState += 2500
                    break
            else:
                nthState += 1
                break


# colone 
#   a  b  c  d  e  f  g  h  i  k (k au lieu de j mais sinon nice)
# 1
# 2
# ...
# 9
# : (au lieu de 10)

# print(stateIntoCharMatrix(state))
# [['F', 'E', 'B', 'M', 'B', 'E', 'F', 'E', 'E', 'F'],
#  ['B', 'G', 'I', 'B', 'H', 'I', 'B', 'E', 'D', 'B'],
#  ['G', 'J', 'D', 'D', 'D', 'H', 'C', 'G', 'J', 'G'],
#  ['D', 'H', 'D', 'L', 'I', 'F', 'K', 'D', 'D', 'H'],
#  ['a', 'a', '_', '_', 'a', 'a', '_', '_', 'a', 'a'],
#  ['a', 'a', '_', '_', 'a', 'a', '_', '_', 'a', 'a'],
#  ['T', 'P', 'P', 'W', 'R', 'U', 'X', 'P', 'T', 'P'],
#  ['S', 'V', 'S', 'O', 'T', 'P', 'P', 'P', 'V', 'S'],
#  ['N', 'P', 'Q', 'N', 'U', 'T', 'N', 'U', 'S', 'N'],
#  ['R', 'Q', 'Q', 'R', 'Q', 'N', 'Y', 'N', 'Q', 'R']]

# print(stateIntoCharMatrix(state.information_state_string()))
# [['F', 'E', 'B', 'M', 'B', 'E', 'F', 'E', 'E', 'F'],
#  ['B', 'G', 'I', 'B', 'H', 'I', 'B', 'E', 'D', 'B'],
#  ['G', 'J', 'D', 'D', 'D', 'H', 'C', 'G', 'J', 'G'],
#  ['D', 'H', 'D', 'L', 'I', 'F', 'K', 'D', 'D', 'H'],
#  ['a', 'a', '_', '_', 'a', 'a', '_', '_', 'a', 'a'],
#  ['a', 'a', '_', '_', 'a', 'a', '_', '_', 'a', 'a'],
#  ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?'],
#  ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?'],
#  ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?'],
#  ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?']]
