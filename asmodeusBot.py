from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import basicAIBot

import pyspiel
import numpy as np
import sys
import random
from statework import *

class asmodeusBot(basicAIBot.basicAIBot):
    def __init__(self, player):
        super().__init__(player)
        if player: # player == 1, the second player
            self.riskScores = {'X' : 0.01 ,'W' : 0.05 ,'V' : 0.15 ,'U' : 0.2, 'T' : 0.2, 'S' : 0.25, 'R' : 0.25,'Q' : 0.01 ,'P' : 0.4, 'O' : 0.01}
            self.bombScores = {'X' : 0.0 , 'W' : 0.0 , 'V' : 0.05 ,'U' : 0.1, 'T' : 0.3, 'S' : 0.4,  'R' : 0.5, 'Q' : 1.0 , 'P' : 0.6, 'O' : 0.1}
            self.flagScores = {'X' : 1.0 , 'W' : 1.0 , 'V' : 1.0 , 'U' : 1.0, 'T' : 1.0, 'S' : 1.0,  'R' : 1.0, 'Q' : 1.0 , 'P' : 1.0, 'O' : 1.0}
            self.suicideScores ={'X':0.0,  'W' : 0.0 , 'V' : 0.0 , 'U' : 0.0, 'T' : 0.0, 'S' : 0.05, 'R' : 0.1, 'Q' : 0.0 , 'P' : 0.0, 'O' : 0.0}
            self.killScores = {'L' : 1.0 , 'K' : 0.9 , 'J' : 0.8 , 'I' : 0.5, 'H' : 0.5, 'G' : 0.5,  'F' : 0.4, 'E' : 0.9 , 'D' : 0.6, 'C' : 0.9}

        else:
            self.riskScores = {'L' : 0.01 ,'K' : 0.05 ,'J' : 0.15 ,'I' : 0.2, 'H' : 0.2, 'G' : 0.25, 'F' : 0.25,'E' : 0.01 ,'D' : 0.4, 'C' : 0.01}
            self.bombScores = {'L' : 0.0 , 'K' : 0.0 , 'J' : 0.05 ,'I' : 0.1, 'H' : 0.3, 'G' : 0.4,  'F' : 0.5, 'E' : 1.0 , 'D' : 0.6, 'C' : 0.1}
            self.flagScores = {'L' : 1.0 , 'K' : 1.0 , 'J' : 1.0 , 'I' : 1.0, 'H' : 1.0, 'G' : 1.0,  'F' : 1.0, 'E' : 1.0 , 'D' : 1.0, 'C' : 1.0}
            self.suicideScores ={'L':0.0,  'K' : 0.0 , 'J' : 0.0 , 'I' : 0.0, 'H' : 0.0, 'G' : 0.05, 'F' : 0.1, 'E' : 0.0 , 'D' : 0.0, 'C' : 0.0}
            self.killScores = {'X' : 1.0 , 'W' : 0.9 , 'V' : 0.8 , 'U' : 0.5, 'T' : 0.5, 'S' : 0.5,  'R' : 0.4, 'Q' : 0.9 , 'P' : 0.6, 'O' : 0.9}

    def __str__(self):
        return "asmodeusBot"

    def make_move(self, policy, state):
        player_pieces = players_pieces()
        partial_state = state.information_state_string(state.current_player())
        matrix = stateIntoCharMatrix(partial_state)
        # print("====================================================")
        # printCharMatrix(partial_state)
        units = []
        enemies = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                piece = matrix[i][j]
                if piece in player_pieces[self.player] and (not piece_protec(matrix, self.player, [i, j], False)) and piece != player_pieces[self.player][0] and piece != player_pieces[self.player][1]:
                    units.append([j, i, piece])
                elif (piece in player_pieces[1-self.player] or piece == '?') and (not piece_protec(matrix, 1-self.player, [i, j], True)):
                    enemies.append([j, i, piece])
        moveList = []
        for unit in units:
            for enemy in enemies:
                path = PathFinder().pathFind((unit[0], unit[1]), (enemy[0], enemy[1]), matrix)
                # print(unit)
                # print(enemy)
                # print(path)
                
                if path == False or len(path) <= 0:
                    continue
                score = self.calculate_score(unit[2], enemy[2])
                score = float(score / float(len(path) + 1))
                moveList.append([unit, path, enemy, score])
        if len(moveList) <= 0:
            # sys.stderr.write("NO Moves!\n")
            return super().make_move(policy, state)
        
        moveList.sort(key = lambda e : e[len(e)-1], reverse=True)
        direction = moveList[0][1][0]
        base = moveList[0][0]
        coord = [base[0], base[1]]
        if direction == "UP":
            coord.append(base[0])
            coord.append(base[1]-1)
        elif direction == "DOWN":
            coord.append(base[0])
            coord.append(base[1]+1)
        elif direction == "LEFT":
            coord.append(base[0]-1)
            coord.append(base[1])
        elif direction == "RIGHT":
            coord.append(base[0]+1)
            coord.append(base[1])
        action_str = coord_to_action(coord)
        actions, proba = np.array(policy).T
        actions = actions.astype(int)
        for action in actions:
            if state.action_to_string(action) == action_str:
                return action
        return action
    
    def spy_attack_ms(self, attacker, defender):
        if self.player:
            return defender == 'L' and attacker == 'O'
        else:
            return defender == 'X' and attacker == 'C'

    def calculate_score(self, attacker, defender):
        player_piece = players_pieces()
        bomb = player_piece[1-self.player][1]
        flag = player_piece[1-self.player][0]
        if defender == '?':
            return self.riskScores[attacker]
        elif defender == bomb:
            return self.bombScores[attacker]
        elif defender == flag:
            return self.flagScores[attacker]
        elif self.valuedRank(defender, 1-self.player) < self.valuedRank(attacker, self.player) or self.spy_attack_ms(attacker, defender):
            return self.killScores[defender]
        else:
            return self.suicideScores[attacker]
        

def piece_protec(matrix, player, coord, ennemy):
    """Check that the piece is surrounded by allies or wall"""
    players_piece = players_pieces()
    protected = True
    to_check = [(coord[0]+1, coord[1]), (coord[0]-1, coord[1]), (coord[0], coord[1]+1), (coord[0], coord[1]-1)]
    for pos in to_check:
        if is_valid_coord(pos):
            if ennemy:
                protected = protected and (matrix[pos[0]][pos[1]] in players_piece[player] or matrix[pos[0]][pos[1]] == "?")
            else:
                protected = protected and (matrix[pos[0]][pos[1]] in players_piece[player])
    return protected


class PathFinder:
	def __init__(self):
		self.visited = []
		pass

	def pathFind(self, start, end, board):
		if start[0] == end[0] and start[1] == end[1]:
			# sys.stderr.write("Got to destination!\n")
			return []

		if self.visited.count(start) > 0:
			# sys.stderr.write("Back track!!\n")
			return False
		if start[0] < 0 or start[0] >= len(board) or start[1] < 0 or start[1] >= len(board[start[0]]):
			# sys.stderr.write("Out of bounds!\n")
			return False
		if len(self.visited) > 0 and board[start[0]][start[1]] != "A" and board[start[0]][start[1]] != "_":
			# sys.stderr.write("Full position!\n")
			return False
		
		self.visited.append(start)
		left = (start[0]-1, start[1])
		right = (start[0]+1, start[1])
		up = (start[0], start[1]-1)
		down = (start[0], start[1]+1)
		choices = [left, right, up, down]
		choices.sort(key = lambda e : (e[0] - end[0])**2.0 + (e[1] - end[1])**2.0 )
		options = []
		for point in choices:
			option = [point, self.pathFind(point,end,board)]
			if option[1] != False:
				options.append(option)	

		options.sort(key = lambda e : len(e[1]))
		if len(options) == 0:
			# sys.stderr.write("NO options!\n")
			return False
		else:
			if options[0][0] == left:
				options[0][1].insert(0,"LEFT")
			elif options[0][0] == right:
				options[0][1].insert(0,"RIGHT")
			elif options[0][0] == up:
				options[0][1].insert(0,"UP")
			elif options[0][0] == down:
				options[0][1].insert(0,"DOWN")
		# sys.stderr.write("PathFind got path " + str(options[0]) + "\n")
		return options[0][1]