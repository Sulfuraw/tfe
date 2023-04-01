# https://github.com/braathwaate/strategoevaluator/blob/master/agents/hunter/hunter.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import basicAIBot
import numpy as np
from statework import *

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
        self.information = updating_knowledge(self.information, state, action)
    
    def make_move(self, policy, state):
        # Can do by scanning too if needed, still 1 for the moment
        # if len(self.units) < 20:
        #     self.maxdepth = 1
        bestMove = self.BestMove(policy, state, self.maxdepth)
        if bestMove == None:
            return super().MakeMove(policy, state)
        
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
        actions, proba = np.array(policy).T
        actions = actions.astype(int)
        # Find the action that correspond to the string format we found
        for action in actions:
            if state.action_to_string(action) == action_str:
                return action
        return action
    
    def PositionLegal(self, x, y, unit = None):
        # TODO: Reverified the calls because it can be "?" too
        if x >= 0 and x < 10 and y >= 0 and y < 10:
            if unit == None:
                return True
            else:
                player_of_unit = self.piece_to_index[unit[2]][1]
                return self.board[x][y] == "A" or self.board[x][y] in self.player_pieces[1-player_of_unit]
                # return self.board[x][y] == None or self.board[x][y].colour == oppositeColour(unit.colour)
        else:
            return False        

    def BestMove(self, policy, state, maxdepth = 1):
        moveList = []

        # if maxdepth < self.maxdepth:
        #     #sys.stderr.write("Recurse!\n")
        #     considerAllies = self.recursiveConsider["allies"]
        #     considerEnemies = self.recursiveConsider["enemies"]
        # else:
        #     considerAllies = len(self.units)+1
        #     considerEnemies = len(self.enemyUnits)+1

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

        for desiredMove in moveList:
            if desiredMove[1] > 0.0:
                desiredMove[1] = desiredMove[1] / float(len(desiredMove[2]))
        
        if len(moveList) <= 0:
            return None
        moveList.sort(key = lambda e : e[1], reverse = True)            
        return moveList[0]
    

    def DesiredMove(self, ally, enemy, matrix):
        """ Determine desired move of allied piece, towards or away from enemy, with score value """
        scaleFactor = 1.0
        # Les ally.rank sont remplacé par des ally[2]
        # 'F' ici est remplacé par self.player_pieces[self.player][0]
        # 'B' pareil mais [1]
        if ally[2] == self.player_pieces[self.player][0] or ally[2] == self.player_pieces[self.player][1]:
            return ["NO_MOVE", 0, None, ally, enemy]
        
        actionScores = {"ATTACK" : 0, "RETREAT" : 0}
        # enemy.rank -> enemy[2]
        if enemy[2] == '?':
            for i in range(0, len(self.ranks[1-self.player])):
                # ranks -> self.ranks[1-self.player]
                prob = self.rankProbability(enemy, self.ranks[1-self.player][i])
                if prob > 0:
                    desiredAction = self.DesiredAction(ally, self.ranks[1-self.player][i])
                    actionScores[desiredAction[0]] += prob * (desiredAction[1] / 2.0)
            # enemy.positions <= 1 -> signifie, cette pièce a jamais bougé
            #                      -> not self.information[2][enemy[0]][enemy[1]]
            # ally.rand != '8' -> ally[3] != self.player_pieces[1-self.player][5]
            if not self.information[2][enemy[0]][enemy[1]] and ally[2] != self.player_pieces[1-self.player][5]:
                # valuedRank(ally.rank) -> self.valuedRank(ally[2], self.player)
                # valuedRank('1') -> self.valuedRank(self.player_pieces[1-self.player][11], 1-self.player)
                scaleFactor *= (1.0 - float(self.valuedRank(ally[2], self.player)) / float(self.valuedRank(self.player_pieces[1-self.player][11], 1-self.player)))**2.0
            elif self.information[2][enemy[0]][enemy[1]] and ally[2] != self.player_pieces[1-self.player][5]:
                scaleFactor *= 0.05
        else:
            desiredAction = self.DesiredAction(ally, enemy[2])
            actionScores[desiredAction[0]] += desiredAction[1]

        desiredAction = sorted(actionScores.items(), key = lambda e : e[1], reverse = True)[0]
        direction = None
        # Before it was like beneath but we exchanged x and y, so inverse 0 and 1 position from this
        # directions = {"RIGHT" : enemy.x - ally.x, "LEFT" : ally.x - enemy.x, "DOWN" : enemy.y - ally.y, "UP" : ally.y - enemy.y}
        directions = {"RIGHT" : enemy[1] - ally[1], "LEFT" : ally[1] - enemy[1], "DOWN" : enemy[0] - ally[0], "UP" : ally[0] - enemy[0]}
        if desiredAction[0] == "RETREAT":
            for key in directions.keys():
                directions[key] = -directions[key]

        while direction == None:
            d = sorted(directions.items(), key = lambda e : e[1], reverse = True)
            p = new_position(ally[0], ally[1], d[0][0])
            if self.PositionLegal(p[0], p[1]) and (matrix[p[0]][p[1]] == "A" or matrix[p[0]][p[1]] == enemy[2]):
                direction = d[0][0]
                scaleFactor *= (1.0 - float(max(d[0][1], 0.0)) / 10.0)**2.0
            else:
                del directions[d[0][0]]
                if len(directions.keys()) <= 0:
                    break 

        if direction == None:
            return ["NO_MOVE", 0, [], ally, enemy]
        return [str(ally[0]) + " " + str(ally[1]) + " " + direction, desiredAction[1], [direction], ally, enemy]


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
        elif target[2] != '?':
            return 0.0
        
        target_rank_index = self.ranks[1-self.player].index(targetRank)
        if self.information[2][target[0]][target[1]]:
            return float(moved_piece(self.information[1])[target_rank_index])
        else:
            return float(no_info_piece(self.information[1])[target_rank_index])
        # total = 0.0
        # for rank in self.ranks:
        #     if rank == 'F' or rank == 'B':
        #         if target.lastMoved < 0:
        #             total += self.hiddenEnemies[rank]
        #     else:
        #         total += self.hiddenEnemies[rank]

        # if total == 0.0:
        #     return 0.0
        # return float(float(self.hiddenEnemies[targetRank]) / float(total))
     


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