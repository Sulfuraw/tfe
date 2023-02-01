import curses

# It is: Fl, Bo, Sp, Sc, Mi, Sg, Lt, Cp, Mj, Co, Ge, Ms
piece = [1/12]*12 # Proba that it is a specific piece
matrix_of_possibilities = [[piece]*10]*10
number_of_piece_left = [1, 6, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1]  # [nbr]*12
player_id = 0
knowledge = [matrix_of_possibilities, number_of_piece_left, player_id]

def generate_state(state, knowledge):
    final = ""
    players_piece = [["M", "C", "K", "L", "J", "I", "F","G", "H", "E", "B", "D"],
                    ["Y", "O", "W", "X", "V", "U", "R", "S", "T", "Q", "N", "P"]]
    player_id = knowledge[3]

    return final

def stateIntoCharMatrix(state):
    stat = str(state).split(" ")[0]
    stat = ' '.join(stat)
    statInLines = []
    for i in range(0, len(stat), 20):
        statInLines.append(stat[i:i+20].rstrip().split(" "))
    return statInLines

def printCharMatrix(charMatrix):
    final = ""
    for line in charMatrix:
        for char in line:
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
                if oldPiece in ["M", "C", "K", "L", "J", "I",
                                "F","G", "H", "E", "B", "D"]:
                    color = curses.color_pair(1)
                elif oldPiece in ["Y", "O", "W", "X", "V", "U",
                                "R", "S", "T", "Q", "N", "P"]:
                    color = curses.color_pair(4)
                else: color = curses.color_pair(7)
                if oldPiece in ["M", "Y"]: piece = "Fl"
                elif oldPiece in ["C", "O"]: piece = "Sp"
                elif oldPiece in ["D", "P"]: piece = "Sc"
                elif oldPiece in ["E", "Q"]: piece = "Mi"
                elif oldPiece in ["F", "R"]: piece = "Sg"
                elif oldPiece in ["B", "N"]: piece = "Bo"
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
