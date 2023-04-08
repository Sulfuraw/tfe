import math
import pyspiel
import time
from statework import *

# class gptBot(pyspiel.Bot):
#     def __init__(self, player):
#         pyspiel.Bot.__init__(self)
#         self.player = player
#         self.max_depth = 5
        # self.player_pieces = players_pieces()
        #                     #Fl,       Bo,     Sp,     Sc,     Mi,     Sg,     Lt,     Cp,     Mj,     Co,     Ge,     Ms]
        # self.player_rank = {'M': 100, 'B': 5, 'C': 7, 'D': 1, 'E': 3, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6, 'K': 7, 'L': 8, 
        #                     'Y': 100, 'N': 5, 'O': 7, 'P': 1, 'Q': 3, 'R': 2, 'S': 3, 'T': 4, 'U': 5, 'V': 6, 'W': 7, 'X': 8, '?': 5}

#     def __str__(self):
#         return "gpt"

#     def step(self, state):
#         printCharMatrix(state)
#         print(self._evaluate_state(state, self.player))
#         _, action = self._alpha_beta_search(state, self.max_depth, -math.inf, math.inf, self.player)
#         return action

#     def _alpha_beta_search(self, state, depth, alpha, beta, maximizing_player_id):
        # if state.is_terminal():
        #     return state.returns()[maximizing_player_id], None
        # if depth == 0:
        #     return self._evaluate_state(state, maximizing_player_id), None

#         current_player_id = state.current_player()
#         is_max_node = current_player_id == maximizing_player_id
#         best_value = -math.inf if is_max_node else math.inf
#         best_action = None

#         for action in state.legal_actions():
#             child_state = state.clone()
#             child_state.apply_action(action)

#             value, _ = self._alpha_beta_search(child_state, depth - 1, alpha, beta, maximizing_player_id)

#             if is_max_node and value > best_value:
#                 best_value = value
#                 best_action = action
#                 alpha = max(alpha, best_value)
#             elif not is_max_node and value < best_value:
#                 best_value = value
#                 best_action = action
#                 beta = min(beta, best_value)

#             if beta <= alpha:
#                 break

#         return best_value, best_action

#     def _evaluate_state(self, state, player):
#         """Returns evaluation on given state."""
#         state_str = str(state).upper()
#         partial_state = state.information_state_string(state.current_player())
#         # matrix = stateIntoCharMatrix(partial_state)
#         matrix = stateIntoCharMatrix(state_str)

#         flag_str_pos = str(state).find(self.player_pieces[player][0])
#         flag = [flag_str_pos//10, flag_str_pos%10]
    
#         flag_str_pos = str(state).find(self.player_pieces[1-player][0])
#         eflag = [flag_str_pos//10, flag_str_pos%10]

#         score = 0
#         for i in range(10):
#             for j in range(10):
#                 piece = matrix[i][j]
#                 if piece not in ["A", "_"]:
#                     if piece in self.player_pieces[player]:
#                         # add rank of player's piece
#                         score += self.player_rank[piece]
#                         # subtract distance of player's piece from opponent's flag
#                         man_distance = abs(i - eflag[0]) + abs(j - eflag[1])/200
#                         score -= man_distance
#                     elif piece in self.player_pieces[player] or piece == "?":
#                         # subtract rank of opponent's piece
#                         score -= self.player_rank[piece]
#                         # add distance of opponent's piece from player's flag: Enlever ça ? 171 moves et c'était finito enfaite mais moves bizarre
#                         # man_distance = abs(i - flag[0]) + abs(j - flag[1])/200
#                         score += man_distance
#         return (score) / (200)

class gptBot(pyspiel.Bot):
    def __init__(self, player, time_limit=3):
        pyspiel.Bot.__init__(self)
        self.player = player
        self.time_limit = time_limit
        self.max_depth = 1
        self.player_pieces = players_pieces()
                            #Fl,       Bo,     Sp,     Sc,     Mi,     Sg,     Lt,     Cp,     Mj,     Co,     Ge,     Ms]
        self.player_rank = {'M': 100, 'B': 5, 'C': 7, 'D': 1, 'E': 3, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6, 'K': 7, 'L': 8, 
                            'Y': 100, 'N': 5, 'O': 7, 'P': 1, 'Q': 3, 'R': 2, 'S': 3, 'T': 4, 'U': 5, 'V': 6, 'W': 7, 'X': 8, '?': 5}

    def __str__(self):
        return "gpt"

    def step(self, state):
        start_time = time.time()
        _, action = None, None
        printCharMatrix(state)
        while time.time() - start_time < self.time_limit:
            value, act = self._alpha_beta_search(state, self.max_depth, -math.inf, math.inf, self.player, start_time)
            if value is not None and act is not None:
                _, action = value, act
            self.max_depth += 1
        return action

    def _alpha_beta_search(self, state, depth, alpha, beta, maxi_player, start_time):
        if state.is_terminal():
            return state.returns()[maxi_player], None
        
        if depth == 0:
            return self._evaluate_state(state, maxi_player), None

        if time.time() - start_time > self.time_limit:
            return None, None
        
        current_player = state.current_player()

        if current_player == self.player:
            max_value = -math.inf
            max_action = None
            for action in state.legal_actions(current_player):
                child_state = state.clone()
                child_state.apply_action(action)
                value, _ = self._alpha_beta_search(child_state, depth - 1, alpha, beta, maxi_player, start_time)
                if value is not None and value > max_value:
                    max_value = value
                    max_action = action
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            return max_value, max_action
        else:
            min_value = math.inf
            min_action = None
            for action in state.legal_actions(current_player):
                child_state = state.clone()
                child_state.apply_action(action)
                value, _ = self._alpha_beta_search(child_state, depth - 1, alpha, beta, maxi_player, start_time)
                if value is not None and value < min_value:
                    min_value = value
                    min_action = action
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            return min_value, min_action

    def _evaluate_state(self, state, player):
        """Returns evaluation on given state."""
        state_str = str(state).upper()
        partial_state = state.information_state_string(state.current_player())
        # matrix = stateIntoCharMatrix(partial_state)
        matrix = stateIntoCharMatrix(state_str)

        flag_str_pos = str(state).find(self.player_pieces[player][0])
        flag = [flag_str_pos//10, flag_str_pos%10]
    
        flag_str_pos = str(state).find(self.player_pieces[1-player][0])
        eflag = [flag_str_pos//10, flag_str_pos%10]

        score = 0
        for i in range(10):
            for j in range(10):
                piece = matrix[i][j]
                if piece not in ["A", "_"]:
                    if piece in self.player_pieces[player]:
                        # add rank of player's piece
                        score += self.player_rank[piece]
                        # subtract distance of player's piece from opponent's flag
                        man_distance = abs(i - eflag[0]) + abs(j - eflag[1])/200
                        score -= man_distance
                    elif piece in self.player_pieces[player] or piece == "?":
                        # subtract rank of opponent's piece
                        score -= self.player_rank[piece]
                        # add distance of opponent's piece from player's flag: Enlever ça ? 171 moves et c'était finito enfaite mais moves bizarre
                        # man_distance = abs(i - flag[0]) + abs(j - flag[1])/200
                        score += man_distance
        return (score) / (200)