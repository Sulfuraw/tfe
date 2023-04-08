# https://github.com/braathwaate/strategoevaluator/blob/master/agents/hunter/hunter.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import basicAIBot
import numpy as np
from statework import *
import time

class hunterBot(basicAIBot.basicAIBot):
    """ Implementation with only a max-depth of 1 
        Because if we want more than max-depth 1 we need
        to be capable of simulation of game with only 
        partial information.
    """
    def __init__(self, player):
        super().__init__(player)
        self.scoreTable = [9, 10, 9, 8, 7, 6, 5, 4, 9, 2, 5, 9]
        self.maxdepth = 1
        self.recursiveConsider = {"allies" : 5, "enemies" : 5}
        self.player_pieces = players_pieces()
        self.piece_to_index = pieces_to_index()

        self.information = []

    def __str__(self):
        return "hunter"
    
    def init_knowledge(self):
        nbr_piece_left = np.array([1, 6, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1])
        moved_before = np.zeros((10, 10))
        moved_scout = np.zeros((10, 10))
        self.information = [self.player, nbr_piece_left, moved_before, moved_scout]

    def update_knowledge(self, state, action):
        updating_knowledge(self.information, state, action)
    
    def make_move(self, policy, state):
        i = 0
        found = False
        bestMoveList = self.BestMove(policy, state, self.maxdepth)
        # Modification to forbid to do the two cases forbidden move: Going back and forth between two positions
        while not found and i < len(bestMoveList):
            bestMove = bestMoveList[i]
            if bestMove == None or bestMove[2] == None:
                return super().make_move(policy, state)
            
            direction = bestMove[2][0] # ['UP'][0] fe
            coord = [bestMove[3][1], bestMove[3][0]] # [6, 5, 'U'][1] or [0] fe  
            if direction == "UP":
                coord.append(coord[0])
                coord.append(coord[1]-1)
            elif direction == "DOWN":
                coord.append(coord[0])
                coord.append(coord[1]+1)
            elif direction == "LEFT":
                coord.append(coord[0]-1)
                coord.append(coord[1])
            elif direction == "RIGHT":
                coord.append(coord[0]+1)
                coord.append(coord[1])
            action_str = coord_to_action(coord)
            # Use the last_moves list of 5 last moves done to take the next best move if the current one participate in the forbidden move
            if action_str == self.last_moves[1] and self.last_moves[1] == self.last_moves[3] and self.last_moves[0] == self.last_moves[2]:
                i += 1
            else:
                found = True
        actions, proba = np.array(policy).T
        actions = actions.astype(int)
        # Find the action that correspond to the string format we found
        for action in actions:
            if state.action_to_string(action) == action_str:
                return action
        return action
    
    def PositionLegal(self, x, y, matrix, unit = None):
        if x >= 0 and x < 10 and y >= 0 and y < 10:
            if unit == None:
                return True
            else:
                player_of_unit = self.piece_to_index[unit[2]][1]
                return matrix[x][y] == "A" or matrix[x][y] in self.player_pieces[1-player_of_unit]
        else:
            return False        

    def BestMove(self, policy, state, maxdepth = 1):
        moveList = []

        partial_state = state.information_state_string(state.current_player())
        matrix = stateIntoCharMatrix(partial_state)
        
        # Locate all moveable ally piece and all accessible enemy piece
        units = []
        enemies = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                piece = matrix[i][j]
                if piece in self.player_pieces[self.player] and (not piece_protec(matrix, self.player, [i, j], False)) and piece != self.player_pieces[self.player][0] and piece != self.player_pieces[self.player][1]:
                    units.append([i, j, piece])
                elif (piece in self.player_pieces[1-self.player] or piece == '?') and (not piece_protec(matrix, 1-self.player, [i, j], True)):
                    enemies.append([i, j, piece])
        # Check for each combination of unit / enemy if there is a path, and compute the score of this
        moveList = []
        for ally in units:
            for enemy in enemies:
                moveList.append(self.DesiredMove(ally, enemy, matrix))

        for desiredMove in moveList:
            if desiredMove[0] == "NO_MOVE" or desiredMove[2] == None:
                desiredMove[1] = -2.0

        # If maxdepth was higher than 1, then this normalization would work
        for desiredMove in moveList:
            if desiredMove[1] > 0.0:
                desiredMove[1] = desiredMove[1] / float(len(desiredMove[2]))
        # But we recalculate it another way (Changes done by me)
        for desiredMove in moveList:
            if desiredMove[1] > 0.0 and desiredMove[6] != 0:
                desiredMove[1] = desiredMove[1] / float(desiredMove[6])
        # Check that it wants to retreat, if it's chosen and doesnt get away from the enemy, then we cancel it
        for desiredMove in moveList:
            if desiredMove[5] == "RETREAT":
                direction = desiredMove[2][0]
                ally = desiredMove[3]
                enemy = desiredMove[4]
                new_pos = new_position(ally[0], ally[1], direction)
                man_distance_before = abs(enemy[0] - ally[0]) + abs(enemy[1] - ally[1])
                man_distance_after = abs(enemy[0] - new_pos[0]) + abs(enemy[1] - new_pos[1])
                if man_distance_before >= man_distance_after or man_distance_before > 3:
                    desiredMove[1] = -2.0
        
        if len(moveList) <= 0:
            return None
        moveList.sort(key = lambda e : e[1], reverse = True)
        # print("=====================================")
        printCharMatrix(state)
        # print(moveList[:10])
        return moveList
    

    def DesiredMove(self, ally, enemy, matrix):
        """ Determine desired move of allied piece, towards or away from enemy, with score value """
        scaleFactor = 1.0
        # [Changes done by me]
        path = PathFinder().pathFind((ally[0], ally[1]), (enemy[0], enemy[1]), matrix)
        man_distance = abs(enemy[0] - ally[0]) + abs(enemy[1] - ally[1])
        if ally[2] == self.player_pieces[self.player][0] or ally[2] == self.player_pieces[self.player][1] or path == False or len(path) <= 0: #or man_distance < len(path)/3:
            return ["NO_MOVE", 0, None, ally, enemy, 'NO_MOVE', 0]
        
        actionScores = {"ATTACK" : 0, "RETREAT" : 0}
        if enemy[2] == '?':
            for i in range(0, len(self.ranks[1-self.player])):
                prob = self.rankProbability(enemy, self.ranks[1-self.player][i])
                if prob > 0:
                    desiredAction = self.DesiredAction(ally, self.ranks[1-self.player][i])
                    actionScores[desiredAction[0]] += prob * (desiredAction[1] / 2.0)
            if not self.information[2][enemy[0]][enemy[1]] and ally[2] != self.player_pieces[1-self.player][4]:
                scaleFactor *= (1.0 - float(self.valuedRank(ally[2], self.player)) / float(self.valuedRank(self.player_pieces[1-self.player][11], 1-self.player)))**2.0
            elif self.information[2][enemy[0]][enemy[1]] and ally[2] == self.player_pieces[1-self.player][4]:
                scaleFactor *= 0.05
        else:
            desiredAction = self.DesiredAction(ally, enemy[2])
            actionScores[desiredAction[0]] += desiredAction[1]

        desiredAction = sorted(actionScores.items(), key = lambda e : e[1], reverse = True)[0]
        direction = None
        directions = {"UP" : ally[0] - enemy[0], "DOWN" : enemy[0] - ally[0], "RIGHT" : enemy[1] - ally[1], "LEFT" : ally[1] - enemy[1]}
        if desiredAction[0] == "RETREAT":
            for key in directions.keys():
                directions[key] = -directions[key]

        while direction == None:
            d = sorted(directions.items(), key = lambda e : e[1], reverse = True)
            p = new_position(ally[0], ally[1], d[0][0])
            if self.PositionLegal(p[0], p[1], matrix) and (matrix[p[0]][p[1]] == "A" or matrix[p[0]][p[1]] == enemy[2]):
                direction = d[0][0]
                scaleFactor *= (1.0 - float(max(d[0][1], 0.0)) / 10.0)**2.0
            else:
                del directions[d[0][0]]
                if len(directions.keys()) <= 0:
                    break 
        if direction == None:
            return ["NO_MOVE", 0, None, ally, enemy, 'NO_MOVE', 0]
        path_direction = path[0]
        # [Changes done by me]
        if desiredAction[0]=="ATTACK" and path_direction != direction:
            # scaleFactor *= 0.5
            return [str(ally[0]) + " " + str(ally[1]) + " " + path_direction, desiredAction[1]*scaleFactor, [path_direction], ally, enemy, desiredAction[0], man_distance, path_direction]
        return [str(ally[0]) + " " + str(ally[1]) + " " + direction, desiredAction[1]*scaleFactor, [direction], ally, enemy, desiredAction[0], man_distance, path_direction]


    def DesiredAction(self, ally, enemyRank):
        enemy_flag = self.player_pieces[1-self.player][0] # 'F'
        enemy_bomb = self.player_pieces[1-self.player][1] # 'B'
        ally_miner = self.player_pieces[self.player][4] # '8'
        enemy_spy = self.player_pieces[1-self.player][2] # 's'
        ally_spy = self.player_pieces[self.player][2] # 's'
        enemy_marshal = self.player_pieces[1-self.player][11] # '1'
        ally_marshal = self.player_pieces[self.player][11] # '1'
        if enemyRank == enemy_flag:
            return ["ATTACK", 1.0]
        if ally[2] == ally_miner and enemyRank == enemy_bomb:
            return ["ATTACK", 0.9]
        if ally[2] == ally_marshal and enemyRank == enemy_spy:
            return ["RETREAT", 0.9]
        if ally[2] == ally_spy and enemyRank == enemy_marshal:
            return ["ATTACK", 0.6]
        if enemyRank == enemy_bomb:
            return ["RETREAT", 0.0]
        if ally[2] == enemyRank:
            return ["ATTACK", 0.1]
        if self.valuedRank(ally[2], self.player) > self.valuedRank(enemyRank, 1-self.player):
            return ["ATTACK", float(self.scoreTable[self.ranks[1-self.player].index(enemyRank)]) * (0.1 + 1.0/float(self.scoreTable[self.ranks[self.player].index(ally[2])]))]
        else:
            return ["RETREAT", float(self.scoreTable[self.ranks[self.player].index(ally[2])]) / 10.0]


    def rankProbability(self, target, targetRank):
        if target[2] == targetRank:
            return 1.0
        if target[2] != '?':
            return 0.0
        # Because of the way we store information, the scout may be in this case too if he moved like that before
        if self.information[3][target[0]][target[1]] and targetRank == self.player_pieces[1-self.player][3]:
            return 1.0
        if not self.information[3][target[0]][target[1]] and targetRank == self.player_pieces[1-self.player][3]:
            return 0.0
        
        target_rank_index = self.ranks[1-self.player].index(targetRank)
        if self.information[2][target[0]][target[1]]:
            return float(moved_piece(self.information[1])[target_rank_index])
        else:
            return float(no_info_piece(self.information[1])[target_rank_index])
 

def new_position(x, y, direction):
    """ Return the new position of a piece after a move 
        x, y : position of the piece
        direction : direction of the move in the str format
    """
    if direction == "UP":
        return [x - 1, y]
    elif direction == "DOWN":
        return [x + 1, y]
    elif direction == "LEFT":
        return [x, y - 1]
    elif direction == "RIGHT":
        return [x, y + 1]
    else:
        return [x, y]
    
def OppositeDirection(direction):
    if direction == "UP":
        return "DOWN"
    elif direction == "DOWN":
        return "UP"
    elif direction == "LEFT":
        return "RIGHT"
    elif direction == "RIGHT":
        return "LEFT"
    else:
        assert(False)
    return "ERROR"

def piece_protec(matrix, player, coord, ennemy):
    """Check that the piece is surrounded by allies or wall"""
    players_piece = players_pieces()
    protected = True
    to_check = [[coord[0]+1, coord[1]], [coord[0]-1, coord[1]], [coord[0], coord[1]+1], [coord[0], coord[1]-1]]
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
        if len(self.visited) > 0 and board[start[0]][start[1]] != "A": #and board[start[0]][start[1]] != "_":
            # sys.stderr.write("Full position!\n")
            return False
        
        self.visited.append(start)
        left = (start[0], start[1]-1)
        right = (start[0], start[1]+1)
        up = (start[0]-1, start[1])
        down = (start[0]+1, start[1])
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
    