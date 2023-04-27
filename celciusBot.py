# Celcius1.1
import random

# He knowns when it has moved, it's not a bomb "!"
# He knowns when it has move like a scout '9'

ranks = ['B','1','2','3','4','5','6','7','8','9','s','F', '?', '!', '+']

"""
The scaretable lists how `scary' pieces are to each other; pieces will move
in the least scary direction.
"""

#	         B   1  2  3  4  5  6  7  8  9  s  F  ?  !  +
scaretable = [[  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #B
              [  0,  0,-8,-8,-7,-6,-5,-4,-3,-2, 5,-9, 0,-7, 0], #1
            [  0,  4, 0,-7,-6,-5,-4,-3,-2,-1,-2,-9,-3,-6, 0], #2
            [  0,  4, 2, 0,-6,-5,-4,-3,-2,-1,-2,-9,-2,-5, 0], #3
            [  0,  3, 2, 2, 0,-5,-4,-3,-2,-1,-2,-9,-1,-3, 0], #4 
            [  0,  3, 2, 2, 2, 0,-4,-3,-2,-1,-2,-9, 0,-2, 0], #5
            [  0,  3, 2, 2, 2, 2, 0,-3,-2,-1,-2,-9, 1,-1, 0], #6
            [  0,  3, 2, 2, 2, 2, 2, 0,-2,-1,-2,-9,-1, 0, 0], #7
            [-40,  3, 2, 2, 2, 2, 2, 2, 0,-2,-2,-9,-1, 1, 0], #8
            [  0,  3, 2, 2, 2, 2, 2, 2, 2, 0,-2,-9,-2, 2, 0], #9
            [  0, -5, 3, 3, 3, 3, 3, 3, 3, 3,-1,-9, 5, 3, 0], #s
            [  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #F
            [  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #?
            [  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #!
            [  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] #+

"""
The override table allows moves to be forced or prevented, thus ensuring
that sacrifices are not made.
"""
#	        B  1  2  3  4  5  6  7  8  9  s  F  ?  !  +
overrides  = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #B
              [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1], #1
            [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1], #2
            [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1], #3
            [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1], #4 
            [ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1], #5
            [ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,-1, 0, 0, 1], #6
            [ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,-1, 0, 0, 1], #7
            [-1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,-1, 0, 0, 1], #8
            [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,-1, 0, 0, 1], #9
            [ 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 0, 0, 1], #s
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #F
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #?
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #!
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] #+

class Piece:
	""" Class representing a piece 
		Pieces have colour, rank and co-ordinates	
	"""
	def __init__(self, colour, rank, x, y):
		self.colour = colour
		self.rank = rank
		self.x = x
		self.y = y
		self.lastMoved = -1
		self.beenRevealed = False
		self.positions = [(x, y)]
		


		self.heatmap = []
		self.turnCount = 0

	def mobile(self):
		return self.rank != 'F' and self.rank != 'B' and self.rank != '?' and self.rank != '+'

	def valuedRank(self):
		if ranks.count(self.rank) > 0:
			return len(ranks) - 2 - ranks.index(self.rank)
		else:
			return 0

	def scariness(self, other):
		scare = scaretable[ranks.index(self.rank)][ranks.index(other.rank)]
		if scare > 0:
			scare = scare * 1
		return scare

	def getOverride(self, other):
		return overrides[ranks.index(self.rank)][ranks.index(other.rank)]

	def getHeatmap(self, x,y,w,h):
		if (x < 0) or (x >= w) or (y < 0) or (y >= h):
			return 10
		else:
			return self.heatmap[x][y]

	def validSquare(self, x, y, width, height, board):
		if x < 0:
			return False
		if y < 0:
			return False
		if x >= width:
			return False
		if y >= height:
			return False
		if board[x][y] != None and board[x][y].colour == self.colour:
			return False
		if board[x][y] != None and board[x][y].rank == '#':
			return False
		return True

	def generateHeatmap(self, width, height, board):
		self.heatmap = []
		newmap = []
		for x in range(0,width):
			self.heatmap.append([])
			newmap.append([])
			for y in range(0,height):
				self.heatmap[x].append(0)
				newmap[x].append(0)
				if board[x][y] == None:
					self.heatmap[x][y] = 0
					continue
				if board[x][y].colour == self.colour:
					if board[x][y].rank == 'F':
						self.heatmap[x][y] = -5 # + self.valuedRank()		# Defend our flag
				else:
					self.heatmap[x][y] = self.scariness(board[x][y])

		# Make pieces prefer to stay where they are
		#self.heatmap[self.x][self.y] = -0.5

		for i in range(0,min(30,len(self.positions))):
			p = self.positions[len(self.positions)-1-i]
			if board[p[0]][p[1]] != None:
				self.heatmap[p[0]][p[1]] += 0.2 * ((50 - i)/50)
				


		for n in range(0,8):
			for x in range(0,width):
				for y in range(0,height):
					if self.heatmap[x][y] != 0:
						newmap[x][y] = self.heatmap[x][y]
						continue
					newmap[x][y] = 0 #self.heatmap[x][y] * 0.2
					if self.validSquare(x-1,y,width,height,board):
						newmap[x][y] += self.heatmap[x-1][y] * 0.2
					else:
						newmap[x][y] += 0 #self.heatmap[x][y] * 0.1
					if self.validSquare(x+1,y,width,height,board):
						newmap[x][y] += self.heatmap[x+1][y] * 0.2
					else:
						newmap[x][y] += 0 #self.heatmap[x][y] * 0.1
					if self.validSquare(x,y-1,width,height,board):
						newmap[x][y] += self.heatmap[x][y-1] * 0.2
					else:
						newmap[x][y] += 0 #self.heatmap[x][y] * 0.1
					if self.validSquare(x,y+1,width,height,board):
						newmap[x][y] += self.heatmap[x][y+1] * 0.2
					else:
						newmap[x][y] += 0 #self.heatmap[x][y] * 0.1
			self.heatmap = newmap

def MakeMove(self):
    """ Randomly moves any moveable piece, or prints "NO_MOVE" if there are none """

    if len(self.units) <= 0:
        return False

    index = random.randint(0, len(self.units)-1)
    startIndex = index

    directions = ("UP", "DOWN", "LEFT", "RIGHT")
    bestdir = 0
    bestScare = 999
    bestpiece = None
    while True:
        piece = self.units[index]

        if piece != None and piece.mobile():
            dirIndex = random.randint(0, len(directions)-1)
            startDirIndex = dirIndex
            piece.generateHeatmap(self.width, self.height, self.board)		
            currentScary = piece.getHeatmap(piece.x, piece.y, self.width, self.height) * 0 + piece.turnCount*0 #Perhaps just look for the best move
            piece.turnCount = piece.turnCount + 1
            while True:
                # Do a move 
                p = move(piece.x, piece.y, directions[dirIndex],1)
                if p[0] >= 0 and p[0] < self.width and p[1] >= 0 and p[1] < self.height:
                    target = self.board[p[0]][p[1]]
                    if target == None or (target.colour != piece.colour and target.colour != "NONE" and target.colour != "BOTH"):	
                        scare = piece.getHeatmap(p[0], p[1],self.width, self.height) - currentScary
                        override = 0
                        if target != None:
                            override = piece.getOverride(target)
                        
                        if (self.total_turns % 250 < 15) and (self.total_turns > 250):
                            scare += random.randint(0, 5)


                        if override == 1:
                            scare = 999
                        elif override == -1:
                            piece.turnCount = 0
                            print(str(piece.x) + " " + str(piece.y) + " " + directions[dirIndex])
                            return True


                        

                        if scare < bestScare:
                            bestdir = dirIndex
                            bestScare = scare
                            bestpiece = piece

                dirIndex = (dirIndex + 1) % len(directions)
                if startDirIndex == dirIndex:
                    break


        index = (index + 1) % len(self.units)
        if startIndex == index:
            if bestScare != 999:
                bestpiece.turnCount = 0
                print(str(bestpiece.x) + " " + str(bestpiece.y) + " "+directions[bestdir])  
                return True
            else:
                print("SURRENDER")
                return True