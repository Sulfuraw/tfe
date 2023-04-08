import curses
import numpy as np
import random

def players_pieces():
    pieces = [["M", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
              ["Y", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"]]
    # It is:  [Fl,  Bo,   Sp,  Sc,  Mi,  Sg,  Lt,  Cp,  Mj,  Co,  Ge,  Ms]
    # Nbr     [1,   6,    1,   8,   5,   4,   4,   4,   3,   2,   1,   1]
    return pieces

def pieces_to_index():
    # In the form [index_of_piece, id_of_player]
    return {"M":[0, 0], "B":[1, 0], "C":[2, 0], "D":[3, 0], "E":[4, 0], "F":[5, 0], "G":[6, 0], "H":[7, 0], "I":[8, 0], "J":[9, 0], "K":[10, 0], "L":[11, 0],
            "Y":[0, 1], "N":[1, 1], "O":[2, 1], "P":[3, 1], "Q":[4, 1], "R":[5, 1], "S":[6, 1], "T":[7, 1], "U":[8, 1], "V":[9, 1], "W":[10, 1], "X":[11, 1]}

matrix_of_stats = [
    [[9.21091620e-02,1.48408795e-01,3.58539765e-03,1.93729999e-01,
    2.69912291e-01,1.40986133e-01,6.33074553e-02,4.39877919e-02,
    2.50237051e-02,1.16006874e-02,2.97795425e-03,4.37062937e-03],
    [2.45644186e-02,2.67912173e-01,3.62984473e-03,1.78455020e-01,
    2.33110110e-01,1.19533009e-01,7.20487140e-02,4.84473154e-02,
    3.66540239e-02,1.18081071e-02,1.94085575e-03,1.89640868e-03],
    [7.22709494e-02,3.00743748e-01,4.57804907e-03,1.10036150e-01,
    2.08590139e-01,1.14969776e-01,7.07153016e-02,6.10258386e-02,
    4.66398009e-02,7.74860732e-03,1.09636127e-03,1.58527913e-03],
    [1.33267157e-01,2.63156335e-01,4.29655091e-03,1.00643001e-01,
    2.23583620e-01,9.51167477e-02,6.61076212e-02,6.14110466e-02,
    4.17061752e-02,8.68199597e-03,7.55600332e-04,1.27414958e-03],
    [9.81391490e-02,2.68919640e-01,6.32630082e-03,1.44290032e-01,
    2.10590257e-01,9.18424796e-02,5.15734266e-02,6.93967050e-02,
    4.56619652e-02,9.67464739e-03,1.25933389e-03,2.32606377e-03],
    [7.60489510e-02,2.76697878e-01,6.65224606e-03,1.47742088e-01,
    2.10990281e-01,1.00983762e-01,5.78256489e-02,6.26111177e-02,
    4.62990399e-02,1.08006400e-02,1.52601636e-03,1.82233021e-03],
    [1.63683774e-01,2.51244518e-01,4.88917862e-03,9.85688041e-02,
    2.12945952e-01,9.19610051e-02,6.56038876e-02,5.58551618e-02,
    4.56767808e-02,7.31895223e-03,1.17043973e-03,1.08154557e-03],
    [8.37679270e-02,2.85735451e-01,5.09659832e-03,9.68946308e-02,
    2.24072538e-01,1.17043973e-01,6.78558729e-02,6.48334716e-02,
    4.45952353e-02,7.79305440e-03,1.22970250e-03,1.08154557e-03],
    [2.23420647e-02,2.79883252e-01,4.68175892e-03,1.55327723e-01,
    2.32473035e-01,1.12406661e-01,7.78416499e-02,6.33222709e-02,
    3.65058670e-02,1.11265853e-02,2.10382838e-03,1.98530283e-03],
    [1.20718265e-01,1.40986133e-01,4.19284106e-03,1.91374304e-01,
    2.64089724e-01,1.37948915e-01,5.14104540e-02,4.41359488e-02,
    3.09499822e-02,8.14863103e-03,2.42977362e-03,3.61502904e-03]],
    [[3.70392320e-03,2.41466161e-01,1.13488207e-02,1.97063530e-01,
    1.44453005e-01,1.22525779e-01,8.56198886e-02,7.97084272e-02,
    5.53810596e-02,3.37945952e-02,9.91169847e-03,1.50231125e-02],
    [1.86677729e-03,1.49505156e-01,1.65935759e-02,2.26502311e-01,
    1.40215716e-01,9.44944886e-02,9.47463553e-02,8.98719924e-02,
    9.17684011e-02,6.20333057e-02,1.42971435e-02,1.81047766e-02],
    [9.92651416e-03,1.89507526e-01,4.71139030e-02,1.22896172e-01,
    1.54260993e-01,7.07449330e-02,8.69977480e-02,1.00272609e-01,
    1.04435818e-01,8.66718028e-02,1.22081309e-02,1.49638497e-02],
    [1.70973095e-02,1.94693019e-01,7.05078819e-02,1.07710087e-01,
    1.67254356e-01,7.55748489e-02,6.83892379e-02,9.11461420e-02,
    1.15429062e-01,7.35895460e-02,7.61526609e-03,1.09932440e-02],
    [2.63719332e-03,1.74988147e-01,4.06542610e-02,1.87359251e-01,
    1.46467939e-01,7.33969420e-02,8.26271186e-02,9.08794595e-02,
    9.25536328e-02,6.99448856e-02,1.80158824e-02,2.04752874e-02],
    [2.74090316e-03,1.60128008e-01,4.31729288e-02,1.86233258e-01,
    1.42156572e-01,6.97670973e-02,7.38414128e-02,9.64797914e-02,
    1.07235984e-01,7.85824345e-02,1.80306981e-02,2.16309115e-02],
    [1.90529809e-02,1.94796729e-01,7.39154913e-02,1.05932203e-01,
    1.69076686e-01,6.76780846e-02,7.31154439e-02,9.16943226e-02,
    1.20496029e-01,6.47445775e-02,9.51167477e-03,9.98577693e-03],
    [1.05932203e-02,1.91892853e-01,5.45661965e-02,1.15266090e-01,
    1.51060804e-01,7.23746592e-02,7.75601517e-02,9.98133223e-02,
    1.09947256e-01,8.80941093e-02,1.25488918e-02,1.62824464e-02],
    [1.76306744e-03,1.40615740e-01,1.99715539e-02,2.28650587e-01,
    1.49964442e-01,9.98874007e-02,9.12498518e-02,9.60353206e-02,
    6.80484769e-02,7.03745407e-02,1.32304137e-02,2.02086050e-02],
    [7.48192485e-03,2.61200664e-01,1.24451819e-02,1.92944767e-01,
    1.37163684e-01,1.15517957e-01,7.75157046e-02,7.29969183e-02,
    6.42556596e-02,3.78244637e-02,1.04006163e-02,1.02524594e-02],]
    ,[[1.49638497e-03,1.10628778e-01,3.49650350e-03,2.34873178e-01,
    5.48625104e-02,1.32541188e-01,1.04346924e-01,1.46245703e-01,
    9.52797203e-02,5.54255067e-02,3.02684604e-02,3.05351428e-02],
    [3.40760934e-04,1.22807277e-01,5.36328079e-03,2.66993600e-01,
    4.72916914e-02,7.68638142e-02,1.12080716e-01,1.39356406e-01,
    9.19017423e-02,6.96930188e-02,3.74096243e-02,2.98980680e-02],
    [1.58527913e-03,1.00420766e-01,7.14412706e-02,1.77047529e-01,
    1.14184544e-01,6.41963968e-02,6.96041247e-02,8.23159891e-02,
    7.74268105e-02,1.16673581e-01,6.43741851e-02,6.07295247e-02],
    [4.62249615e-03,7.29524713e-02,1.20258978e-01,1.88677848e-01,
    1.12628897e-01,5.71737584e-02,8.03158706e-02,7.93380348e-02,
    6.51742325e-02,9.80650705e-02,5.49662202e-02,6.58261230e-02],
    [4.44470783e-04,8.14714946e-02,1.60898424e-02,2.80327723e-01,
    4.52767571e-02,5.97665047e-02,1.10465805e-01,1.31578168e-01,
    1.01991229e-01,6.45519735e-02,5.92331397e-02,4.88028920e-02],
    [4.44470783e-04,7.58119000e-02,1.60157639e-02,2.44118170e-01,
    4.45063411e-02,7.11153254e-02,1.06895223e-01,1.46008652e-01,
    1.02406069e-01,7.15301648e-02,7.73527320e-02,4.37951879e-02],
    [2.57793054e-03,7.20931611e-02,1.26481569e-01,1.89877919e-01,
    1.15651298e-01,5.26549721e-02,7.72934692e-02,7.60045040e-02,
    7.60341354e-02,8.46716842e-02,6.45667891e-02,6.20925684e-02],
    [1.86677729e-03,9.52352732e-02,5.73663625e-02,1.70958279e-01,
    1.12214057e-01,6.00776342e-02,7.50711153e-02,8.79015053e-02,
    7.34858362e-02,1.30052151e-01,6.91003911e-02,6.66706175e-02],
    [3.11129548e-04,1.22644305e-01,6.94855991e-03,2.74875548e-01,
    5.45069337e-02,8.08640512e-02,9.30721821e-02,1.29503971e-01,
    9.55464027e-02,6.48038402e-02,4.19728577e-02,3.49502193e-02],
    [5.42254356e-03,1.09858362e-01,4.07431551e-03,2.19301885e-01,
    5.60625815e-02,1.31207775e-01,1.19592272e-01,1.38141519e-01,
    8.82274505e-02,6.36630319e-02,3.48316937e-02,2.96165699e-02],]
    ,[[1.92604006e-03,8.94423373e-02,6.81521868e-04,3.48331753e-01,
    1.84751689e-02,1.66172810e-01,1.78069812e-01,1.22318360e-01,
    5.12771127e-02,1.27859429e-02,4.82991585e-03,5.68922603e-03],
    [2.37051085e-04,5.91442456e-02,5.03733555e-04,2.98091739e-01,
    1.48749556e-02,1.69773024e-01,2.29287661e-01,1.48897712e-01,
    5.34846509e-02,1.19118170e-02,7.49674055e-03,6.29666943e-03],
    [2.75571886e-03,7.00041484e-02,2.06086287e-02,2.06634467e-01,
    9.55760341e-02,6.98708072e-02,9.99614792e-02,1.09473154e-01,
    1.16080953e-01,8.80052151e-02,5.16771364e-02,6.93522579e-02],
    [2.26680100e-03,5.80330686e-02,4.01801588e-02,1.96500533e-01,
    7.97825056e-02,6.37074790e-02,8.87904468e-02,1.16066137e-01,
    1.03443167e-01,8.31160365e-02,8.46864999e-02,8.34271661e-02],
    [5.33364940e-04,4.48174707e-02,4.74102169e-04,2.95247126e-01,
    1.58083442e-02,1.48201375e-01,2.13108925e-01,1.80025483e-01,
    7.40488325e-02,1.34674647e-02,6.57816760e-03,7.68934455e-03],
    [2.96313856e-04,5.95442693e-02,1.06672988e-03,3.14240844e-01,
    1.42230651e-02,1.43075145e-01,2.10679151e-01,1.69521157e-01,
    6.04332109e-02,1.34526490e-02,6.45964205e-03,7.00782269e-03],
    [1.97048714e-03,6.26703805e-02,4.20469361e-02,1.91181700e-01,
    8.14122318e-02,7.11153254e-02,8.15603888e-02,1.10465805e-01,
    1.14303070e-01,7.27006045e-02,8.47161313e-02,8.58569397e-02],
    [4.96325708e-03,6.79299514e-02,2.73201375e-02,1.83581249e-01,
    9.12054048e-02,6.79003200e-02,9.21536091e-02,1.12036269e-01,
    1.19518194e-01,9.86576982e-02,6.41371341e-02,7.05967761e-02],
    [5.18549247e-04,5.27142349e-02,5.18549247e-04,2.94358184e-01,
    1.51416380e-02,1.64068982e-01,2.25702264e-01,1.59105725e-01,
    6.06109992e-02,1.20896053e-02,7.23005808e-03,7.94121133e-03],
    [1.64454190e-03,8.92941804e-02,8.44494489e-04,3.37160721e-01,
    2.38828968e-02,1.64898661e-01,1.75995615e-01,1.31770772e-01,
    5.00622259e-02,1.43267749e-02,5.85219865e-03,4.26691952e-03]]
]

def no_info_piece(nbr_piece_left):
    """Proba if we have no info"""
    return nbr_piece_left/np.sum(nbr_piece_left)

def moved_piece(nbr_piece_left):
    """Proba if we know this piece moved before"""
    proba = np.array([0.0]*12)
    proba[2:] = nbr_piece_left[2:]/np.sum(nbr_piece_left[2:])
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

def coord_to_action(coord):
    cols = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h", 8:"i", 9:"k"}
    rows = {0:"1", 1:"2", 2:"3", 3:"4", 4:"5", 5:"6", 6:"7", 7:"8", 8:"9", 9:":"}
    return cols[coord[0]] + rows[coord[1]] + cols[coord[2]] + rows[coord[3]]

def is_valid_state(state_str):
    """Verify that the state_str is a valid state, to be used for our generate_state"""
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
        and state_str.count("?") == 0
        and len(state_str) >= 104)

def create_initial_state_str():
    player_pieces = players_pieces()
    setups = [
        [5, 4, 1, 0, 1, 4, 5, 4, 4, 5, 1, 6, 8, 1, 7, 8, 1, 4, 3, 1, 6, 9, 3, 3, 3, 7, 2, 6, 9, 6, 3, 7, 3, 11, 8, 5, 10, 3, 3, 7], # Base p1 Full protec, well spread
        [3, 8, 0, 7, 5, 5, 4, 1, 7, 3, 9, 1, 6, 4, 1, 1, 2, 4, 1, 5, 8, 3, 11, 3, 5, 9, 3, 10, 3, 8, 6, 7, 3, 4, 6, 1, 3, 4, 7, 6], # Base p2 0 protec, well spread
        [6, 1, 4, 1, 5, 3, 7, 0, 7, 4, 3, 5, 1, 4, 1, 7, 8, 8, 3, 3, 5, 3, 8, 1, 9, 10, 4, 3, 9, 6, 3, 4, 6, 3, 1, 6, 2, 11, 7, 5], # 95 moves bait flag, stacked near flag
        [5, 4, 1, 1, 0, 5, 5, 1, 3, 5, 1, 7, 9, 4, 3, 2, 4, 6, 3, 1, 7, 6, 10, 11, 3, 6, 9, 4, 3, 6, 1, 3, 8, 4, 7, 3, 8, 8, 3, 7], # 106 moves 1/3 protec, stacked
        [5, 7, 0, 1, 7, 3, 1, 3, 3, 4, 1, 4, 1, 2, 4, 8, 4, 3, 4, 5, 9, 7, 8, 6, 11, 5, 3, 10, 3, 1, 6, 3, 6, 3, 8, 6, 9, 1, 7, 5], # 130 moves 2/3 protec, not well placed, doubt
        [4, 5, 3, 4, 3, 1, 0, 4, 3, 4, 3, 3, 1, 1, 6, 1, 8, 8, 3, 9, 6, 5, 1, 1, 5, 5, 2, 7, 3, 10, 7, 6, 3, 7, 4, 7, 9, 11, 6, 8], # 150 moves 1/3 protec, not well placed, doubt but less than above
        [6, 7, 1, 3, 1, 3, 0, 1, 3, 1, 9, 4, 4, 2, 4, 3, 1, 8, 11, 7, 4, 6, 8, 10, 9, 6, 5, 3, 5, 8, 5, 3, 7, 3, 6, 1, 4, 3, 7, 5], # 150 moves 2/3 protec, well spread
        [1, 1, 0, 5, 3, 1, 4, 4, 1, 3, 8, 8, 3, 3, 3, 3, 3, 6, 1, 6, 6, 6, 10, 4, 7, 11, 2, 1, 8, 3, 5, 7, 4, 9, 7, 5, 9, 4, 5, 7], # 270 moves 1/3 protec, well spread
        [1, 4, 1, 4, 0, 3, 1, 5, 6, 1, 1, 3, 4, 3, 8, 3, 4, 8, 3, 1, 3, 7, 9, 2, 7, 8, 3, 4, 7, 9, 6, 5, 6, 10, 5, 6, 11, 7, 3, 5], # 340 moves No protec flag, but well spread
        [7, 3, 4, 1, 0, 1, 4, 1, 5, 1, 9, 2, 4, 6, 1, 6, 4, 1, 7, 5, 8, 10, 3, 9, 4, 3, 11, 8, 3, 7, 5, 3, 7, 3, 6, 8, 3, 6, 5, 3]  # 625 moves Full protec flag, well spread
    ]
    setup0 = "".join([player_pieces[0][x] for x in random.choice(setups)])
    setup1 = "".join([player_pieces[1][x] for x in random.choice(setups)[::-1]]) 
    return setup0 + "AA__AA__AAAA__AA__AA" + setup1 + " r 0"

def generate_state(state, information):
    """
        Generate a valid state, with the knowledge of our bot + the partial state
        Information : [self.player_id, nbr_piece_left, moved_before, moved_scout]
    """
    player = state.current_player()
    partial_state = state.information_state_string(player)
    piece_left = information[1].copy()
    moved_before = information[2]
    moved_scout = information[3]
    players_piece = players_pieces()

    state_list = list(partial_state)

    unk_coord = []
    for i in range(100):
        if state_list[i] == "?":
            unk_coord.append(i)

    if player:
        unk_coord = unk_coord[::]
    
    # Treat the scout and moved piece first
    last_unk_coord = []
    for i in unk_coord:
        if moved_scout[i//10][i%10]:
            proba = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif moved_before[i//10][i%10]:
            proba = moved_piece(piece_left)
        else:
            last_unk_coord.append(i)
            continue
        piece_id = np.random.choice(np.arange(12), p=proba)
        if not moved_scout[i//10][i%10]:
            piece_left[piece_id] -= 1
        piece = players_piece[1-player][piece_id]
        state_list[i] = piece
    
    # Comeback on the one we didn't treat: the one that never moved
    for i in last_unk_coord:
        proba = no_info_piece(piece_left)
        piece_id = np.random.choice(np.arange(12), p=proba)
        piece_left[piece_id] -= 1
        piece = players_piece[1-player][piece_id]
        state_list[i] = piece
    
    state_str = "".join(state_list)
    return state_str

def updating_knowledge(information, state, action):
    """
        Each move a player does can affect the information / knowledge that our bot has
        Here we update the variable information
        Information : [self.player_id, nbr_piece_left, moved_before, moved_scout]
    """
    player_id, nbr_piece_left, moved_before, moved_scout = information
    players_piece = players_pieces()
    current_player = state.current_player()

    coord = action_to_coord(state.action_to_string(current_player, action))

    full_matrix_before = stateIntoCharMatrix(state)
    matrix_before = stateIntoCharMatrix(state.information_state_string(player_id))
    start = matrix_before[coord[1]][coord[0]]
    arrival = matrix_before[coord[3]][coord[2]]

    # This is ok because the state passed in argument was a clone
    state.apply_action(action)
    matrix_after = stateIntoCharMatrix(state.information_state_string(player_id))
    arrival_after = matrix_after[coord[3]][coord[2]]

    if current_player == player_id:
        # Fight
        if arrival == "?":
            # Either win or lose the fight, it will result in the same operation
            # We also delete information of moved piece at this place because it doesnt matter anymore
            for p in range(12):
                if full_matrix_before[coord[3]][coord[2]] == players_piece[1-player_id][p]:
                    if not (p==3 and moved_scout[coord[3]][coord[2]]):
                        nbr_piece_left[p] -= 1
                    moved_before[coord[3]][coord[2]] = 0
                    moved_scout[coord[3]][coord[2]] = 0
                    # return [player_id, nbr_piece_left, moved_before, moved_scout]
    else:
        # Fight
        if arrival in players_piece[player_id]:
            was_scout = moved_scout[coord[1]][coord[0]]
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 0
            moved_scout[coord[1]][coord[0]] = 0
            moved_scout[coord[3]][coord[2]] = 0
            # Ennemy Won: But if he killed our piece, we get information of it
            if arrival_after in players_piece[current_player]:
                if start == "?":
                    for p in range(12):
                        if arrival_after == players_piece[current_player][p]:
                            if not (p==3 and was_scout):
                                nbr_piece_left[p] -= 1
            # Ennemy lose:
            else:
                for p in range(12):
                    if full_matrix_before[coord[1]][coord[0]] == players_piece[current_player][p] and start == "?":
                        if not (p==3 and was_scout):
                            nbr_piece_left[p] -= 1
            # return [player_id, nbr_piece_left, moved_before, moved_scout]
        # Deplacement on empty space: Deplacement of moved and/or get information about moved/scout_moved
        else:
            moved_before[coord[1]][coord[0]] = 0
            moved_before[coord[3]][coord[2]] = 1
            if abs(coord[1]-coord[3]) > 1 or abs(coord[0]-coord[2]) > 1:
                if not moved_scout[coord[1]][coord[0]] and start == "?":
                    nbr_piece_left[3] -= 1
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1
            if moved_scout[coord[1]][coord[0]]:
                moved_scout[coord[1]][coord[0]] = 0
                moved_scout[coord[3]][coord[2]] = 1
            # return [player_id, nbr_piece_left, moved_before, moved_scout]
    # return information

def unknown_acc(state, state2):
    """Check the accuracy for the pieces unknown only by comparing the two states piece-wise"""
    str_partial = str(state.information_state_string(state.current_player()))
    str_state1 = str(state).upper()[:100]
    str_state2 = str(state2).upper()[:100]
    total = 0
    good = 0
    for i in range(len(str_state1)):
        if str_partial[i] == "?":
            total += 1
            if str_state1[i] == str_state2[i]:
                good += 1
    return good/total

def global_accuracy_state(state1, state2):
    """Check the accuracy for the pieces (ally & ennemy, known or unkown) by comparing the two states piece-wise"""
    str_state1 = str(state1).upper()[:100]
    str_state2 = str(state2).upper()[:100]
    total = 0
    good = 0
    for i in range(len(str_state1)):
        if str_state1[i] != "_" and str_state1[i] != "A":
            total += 1
            if str_state1[i] == str_state2[i]:
                good += 1
    return good/total

def data_for_games(move, state, generated, evaluator):
    """
        Prepare the data to be saved for a move in a single game
        Move: The number of the move we're at
        Eval_real: The evaluate function used on the real state
        Eval_generated: The evaluate function used on the generated state
        Unknow_acc: The accuracy of prediction of generated regarding the unknown pieces only ("?")
        Unk_pieces: The number of ("?")
        Our_pieces: The number of pieces we have still
        Ennemy_pieces: The number of pieces the ennemy has still ("?" and known pieces)
    """
    player = state.current_player()
    players_piece = players_pieces()
    eval_real = round(evaluator.evaluate(state)[player], 4)
    eval_generated = round(evaluator.evaluate(generated)[player], 4)
    unknow_acc = round(unknown_acc(state, generated), 2)
    unk_pieces = str(state.information_state_string(player)).count("?")
    our_pieces = 0
    ennemy_pieces = 0
    for i in str(generated)[:100].upper():
        if i in players_piece[player]:
            our_pieces += 1
        if i in players_piece[1-player]:
            ennemy_pieces += 1
    # print("====================================")
    # print("move:", move)
    # print("state:")
    # printCharMatrix(state)
    # print("evaluate:", eval_real)
    return {'move': [move], 'eval_real': [eval_real], 'eval_generated': [eval_generated], 'unknow_acc': [unknow_acc], 'unk_pieces': [unk_pieces], 'our_pieces': [our_pieces], 'ennemy_pieces': [ennemy_pieces]}

def is_valid_coord(pos):
    """Tools to check if a position is playable (in board and not a river)"""
    x, y = pos[0], pos[1]
    if x < 0 or x > 9 or y < 0 or y > 9:
        return False
    if y == 2 or y == 3 or y == 6 or y == 7:
        return x != 4 and x != 5
    return True

def flag_protec(state, player):
    """Check that the flag is protected all around him (4 positions)"""
    players_piece = players_pieces()
    matrix = stateIntoCharMatrix(state)
    coord = [0, 0]
    for i in range(10):
        for j in range(10):
            if matrix[i][j] == players_piece[player][0]:
                coord = [i, j]
    protected = True
    to_check = [[coord[0]+1, coord[1]], [coord[0]-1, coord[1]], [coord[0], coord[1]+1], [coord[0], coord[1]-1]]
    for pos in to_check:
        if is_valid_coord(pos):
            protected = protected and (matrix[pos[0]][pos[1]] in players_piece[player])
    return protected 
    
def stateIntoCharMatrix(state):
    """Transform a state into a Matrix of Character inside the str(state)"""
    stat = str(state).upper().split(" ")[0]
    stat = ' '.join(stat)
    statInLines = []
    for i in range(0, len(stat), 20):
        statInLines.append(stat[i:i+20].rstrip().split(" "))
    return statInLines

def printCharMatrix(state):
    """Print device for the matrix of character generated here: stateIntoCharMatrix(state)"""
    charMatrix = stateIntoCharMatrix(state)
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
    """Special printer using an alternative show, using wrapper too"""
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
