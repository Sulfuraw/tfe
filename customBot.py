from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import numpy as np
import math
from statework import *

# Using the Copyright 2019 DeepMind Technologies Limited using modified version of """Monte-Carlo Tree Search algorithm for game play"""
class CustomEvaluator():
    def __init__(self, n_rollouts, n_moves_before):
        self.piece_to_index = pieces_to_index()
        self.player_pieces = players_pieces()
        self.n_rollouts = n_rollouts
        self.n_moves_before = n_moves_before
        self.stateHashPrior = {}
        self.stateHashEvaluate = {}
        self.hash_count = [0, 0, 0, 0]

    def set_information(self, information, last_moves):
        self.information = information
        self.last_moves = last_moves
        self.stateHashPrior = {}
        self.stateHashEvaluate = {}
        self.hash_count = [0, 0, 0, 0]

    def hash_state(self, state):
        state_str = str(state)
        return state_str[:100].upper() + state_str[101]

    # TODO: Make more weight for information we have
    # TODO: Make more weight for information they have
    def evaluate_state(self, state):
        """Returns evaluation on given state."""
        state_str = str(state)[:100].upper()
        move_of_state = int(str(state)[103-len(str(state)):])

        nbr_pieces = [[0]*12, [0]*12]
        advanced_pieces = [0, 0]
        advanced_miner = [0, 0]
        place = 0
        for piece in state_str:
            if piece not in ["A", "_"]:
                i, player = self.piece_to_index[piece]
                nbr_pieces[player][i] += 1
                if player and place < 40:
                    advanced_pieces[player] += 1
                    if i == 4:
                        advanced_miner[player] += 1
                if not player and place > 59:
                    advanced_pieces[player] += 1
                    if i == 4:
                        advanced_miner[player] += 1
            place += 1

        score = [0, 0]
        for player in [0, 1]:
            piece_values = value_for_piece(nbr_pieces[1-player], nbr_pieces[player])
            for piece_id, value in piece_values:
                score[player] += nbr_pieces[player][piece_id]*value
            
            # Make more weight if the flag is protected, 1/4 per side
            score[player] += (0.5/4) * (4 - flag_protec(state, player))

            # More weight having piece on the other side
            score[player] += advanced_pieces[player] * 0.02

            # Make more weight to miner to go attack in search of bombs
            score[player] += advanced_miner[player] * 0.02

        returns = [0, 0]
        for player in [0, 1]:
            x = score[player] - score[1-player]
            # returns[player] = 2*(x - (-8.5))/(8.5 - (-8.5)) - 1.0     # Rerange -8.5 to 8.5 into -1 to 1
            # returns[player] = 2*(x - (-5))/(5 - (-5)) - 1.0           # Value for piece advanced array
            returns[player] = 2*(x - (-9.2))/(9.2 - (-9.2)) - 1.0             # Value for best result so far

        if returns[0] > 1 or returns[0] < -1 or returns[1] > 1 or returns[1] < -1:
            # Check if we need to adapt the rescale above
            print("==============================")
            print("Returns:", returns, "\n")
            printCharMatrix(str(state))
        return returns

    def evaluate(self, state):
        """Returns evaluation on given state by doing simulations."""

        result = None
        # move_of_state = int(str(state)[103-len(str(state)):])
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            i = 0
            while not working_state.is_terminal() and i < self.n_moves_before:
                # We simulate moves with the prior function for both
                legal_actions = self.prior(working_state, i)
                actions, proba = list(zip(*legal_actions))
                action = np.random.choice(actions, p=proba)
                working_state.apply_action(action)
                i += 1
            returns = np.array(working_state.returns())*3 if working_state.is_terminal() else np.array(self.evaluate_state(working_state))
            result = returns if result is None else result + returns
        result = result / self.n_rollouts
        return result
    
    # TODO: Multiply by a riskScore if we gain information on unknown [0, 0, 0.1, 10, 1, 5, 4, 3, 2, 0.3, 0.2, 0.1] ?
    # TODO: Multiply by the value of the piece that win/lose. 
    def proba_win_combat(self, allyIdx, enemy_pos, state, move_of_state):
        _, pieces_left, moved, scout, matrix_of_stats = self.information
        player = state.current_player()
        # Sure of the enemy piece
        if sum(matrix_of_stats[enemy_pos[0]][enemy_pos[1]]) < 0.2:
            return win_combat(allyIdx, str(state)[enemy_pos[0]*10+enemy_pos[1]].upper())
        # Sure it's a scout
        elif scout[enemy_pos[0]][enemy_pos[1]]:
            return win_combat(allyIdx, self.player_pieces[1-player][3])
        # Know it moved before
        elif moved[enemy_pos[0]][enemy_pos[1]]:
            probas = moved_piece_matrix(pieces_left, matrix_of_stats[enemy_pos[0]][enemy_pos[1]])
        # Know it didn't move before
        else:
            probas = no_info_piece_matrix(pieces_left, matrix_of_stats[enemy_pos[0]][enemy_pos[1]])
        summ = 0
        for idx in range(12):
            summ += probas[idx]*win_combat(allyIdx, self.player_pieces[1-player][idx])

        # RiskScore if not fully known only for big pieces to not suicide important on bombs
        if allyIdx > 8 and not moved[enemy_pos[0]][enemy_pos[1]] and summ > 0 and (move_of_state < 200 or probas[1] > 0.2): # or probas[1] > 0.4 # for after the move 350
            return -1
        return summ

    # TODO: Make so that after 700 moves every moves needs to go toward the enemy flag with a search unlimited to avoid being stopped by a wall
    # TODO: Limit moving high piece if not necessary because they run left right next to pieces they cannot attack because of RiskScore
    # TODO: Limit moving Marshal till the enemy spy is discovered or if big piece to take / defend
    # TODO: Search: check all pieces in a range of 3-4 instead of just 1
    # TODO: Search: alternative if the allyIdx == 4 (miner), to go in search / direction of unmoved pieces. check the proba or sum of proba to decide that
    # TODO: Search: alternative for big pieces like 9, 10, 11 to search all pieces and make a better choice. Don't go toward a piece that can kill him (not specially the first we see)
    # TODO: Flee only in a direction that get us away from the enemy, manhanttan distance
    # TODO: More weight if goes toward the flag
    # TODO: Keep only one move with same piece same direction (scout) because they spam the probabilities otherwise ? But we lose the diversity and potential better moves that temporize the game
    # TODO: When we authorized to move to a dangerous place when the place it's on is already dangerous, it should be only if there is unkown piece to discover there. Otherwise it's useless.
    def prior(self, state, where):
        """Returns probability for each actions"""
        self.hash_count[2] += 1
        if self.hash_state(state) in self.stateHashPrior and not (self.last_moves[0] == self.last_moves[2] and self.last_moves[1] == self.last_moves[3]):
            self.hash_count[3] += 1
            return self.stateHashPrior[self.hash_state(state)]

        actions = []
        proba = []
        player = state.current_player()
        state_str = str(state)[:100].upper()
        charMatrix = stateIntoCharMatrix(state)
        move_of_state = int(str(state)[103-len(str(state)):])

        Ms_str_pos = str(state).find(self.player_pieces[1-player][11])
        Ms = [Ms_str_pos//10, Ms_str_pos%10]

        registered_find = {}
        two_square_rule = False
        for action in state.legal_actions(player):

            coord = action_to_coord(state.action_to_string(player, action))
            arrival = state_str[coord[3]*10 + coord[2]]
            allyIdx, _ = self.piece_to_index[state_str[coord[1]*10 + coord[0]]]
            # 2 squares rules: No two repeating move again
            if action == self.last_moves[1] and self.last_moves[0] == self.last_moves[2] and self.last_moves[1] == self.last_moves[3]:
                two_square_rule = True
                value = -30
            # Don't go if dangerous place unless your actual place is also dangerous
            elif dangerous_place([coord[3], coord[2]], charMatrix, allyIdx, player) and not dangerous_place([coord[1], coord[0]], charMatrix, allyIdx, player):
                value = -30
            # Fight
            elif arrival != "A":
                # The scout needs to scout new pieces. 
                if allyIdx == 3 and sum(self.information[4][coord[3]][coord[2]]) > 0.2:
                    value = 15
                else:
                    value = self.proba_win_combat(allyIdx, [coord[3], coord[2]], state, move_of_state)*30
            # Not move the spy till the marshal is near
            # Encourage the movement toward Marshal otherwise: Manhattan distance so not 100% safe
            elif allyIdx == 2:
                Ms_dist_before = (abs(Ms[0] - coord[1]) + abs(Ms[1] - coord[0]))
                if Ms_dist_before < 4 and (abs(Ms[0] - coord[3]) + abs(Ms[1] - coord[2])) < Ms_dist_before:
                    value = 15
                else:
                    value = -30
            # Not move the scout for more than 1 case too fast in the game.
            elif move_of_state < 150 and (abs(coord[1]-coord[3]) > 1 or abs(coord[0]-coord[2]) > 1):
                value = -30
            # Not move the miner too early in the game
            elif move_of_state < 200 and (allyIdx == 4):
                value = -30
            else:
                if (coord[1], coord[0]) in registered_find:
                    found, path = registered_find[(coord[1], coord[0])]
                else:
                    limit = 5
                    found, path = find_combat(charMatrix, (coord[1], coord[0]), self.player_pieces[1-player], limit)
                    registered_find[(coord[1], coord[0])] = [found, path]

                if found:
                    fight_value = self.proba_win_combat(allyIdx, path[-1], state, move_of_state)*30
                # Move toward an enemy piece we win versus, Ally go in the direction of the first enemy
                if found and path[0] == (coord[3], coord[2]) and fight_value >= 0:
                    value = fight_value/np.sqrt(len(path))  # (len(path))
                # Do the same with direction to flee. Flee only big lose
                elif found and path[0] != (coord[3], coord[2]) and fight_value < -9:
                    value = (-fight_value)/np.sqrt(len(path))   # (len(path))
                    # Reduce the value of fleeing for len(path) > 2
                    if len(path) > 2:
                        value = value/2
                else:
                    # Take moving toward the ennemy flag
                    towards_flag = toward_flag(state, player, coord)
                    value = towards_flag[1] if towards_flag[0] else 0

            if dangerous_place([coord[1], coord[0]], charMatrix, allyIdx, player) and value < 0 and not two_square_rule:
                # If sure to loose and doesn't discover anything new, don't change the weight
                if arrival != "A" and sum(self.information[4][coord[3]][coord[2]]) < 0.2:
                    pass
                else:
                    value = 15
            actions.append(action)
            proba.append(value)

        proba = rescaleProbas(proba, 5)
        # Remove completly actions with 0 probability so that it is not selected at all in the mcts search
        mask = proba != 0
        actions = np.array(actions)[mask]
        proba = proba[mask]

        prio = []
        for i in range(len(proba)):
            prio.append([actions[i], proba[i]])

        self.stateHashPrior[self.hash_state(state)] = prio
        return prio

class SearchNode(object):
    """A node in the search tree.

    A SearchNode represents a state and possible continuations from it. Each child
    represents a possible action, and the expected result from doing so.

    Attributes:
        action: The action from the parent node's perspective. Not important for the
        root node, as the actions that lead to it are in the past.
        player: Which player made this action.
        prior: A prior probability for how likely this action will be selected.
        explore_count: How many times this node was explored.
        total_reward: The sum of rewards of rollouts through this node, from the
        parent node's perspective. The average reward of this node is
        `total_reward / explore_count`
        outcome: The rewards for all players if this is a terminal node or the
        subtree has been proven, otherwise None.
        children: A list of SearchNodes representing the possible actions from this
        node, along with their expected rewards.
    """
    __slots__ = [
        "action",
        "player",
        "prior",
        "explore_count",
        "total_reward",
        "outcome",
        "children",
    ]

    def __init__(self, action, player, prior):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

    def uct_value(self, parent_explore_count, uct_c):
        """Returns the UCT value of child."""
        if self.outcome is not None:
            return self.outcome[self.player]

        if self.explore_count == 0:
            return float("inf")

        # MODIFIED HERE, The left part was always way smaller not matter the value of uct_c [0, 1] because the returns are in [-1, 1]
        # The modification switch the left part bound from [-1, 1] (what we have in returns) to [0, 1] (what they are waiting for in this function)
        # print("win/game, right part:", (self.total_reward / self.explore_count), ((self.total_reward / self.explore_count)/2 + 1/2), uct_c*math.sqrt(math.log(parent_explore_count)/self.explore_count))
        return ((self.total_reward / self.explore_count)/2 + 1/2 + 
            uct_c * math.sqrt(math.log(parent_explore_count) / self.explore_count))

    def puct_value(self, parent_explore_count, uct_c):
        """Returns the PUCT value of child."""
        if self.outcome is not None:
            return self.outcome[self.player]

        return ((self.explore_count and self.total_reward / self.explore_count) +     # (self.explore_count and self.total_reward / self.explore_count)/2 + 1/2 +
                uct_c * self.prior * math.sqrt(parent_explore_count) / (self.explore_count + 1))

    def sort_key(self):
        """Returns the best action from this node, either proven or most visited.

        This ordering leads to choosing:
        - Highest proven score > 0 over anything else, including a promising but
        unproven action.
        - A proven draw only if it has higher exploration than others that are
        uncertain, or the others are losses.
        - Uncertain action with most exploration over loss of any difficulty
        - Hardest loss if everything is a loss
        - Highest expected reward if explore counts are equal (unlikely).
        - Longest win, if multiple are proven (unlikely due to early stopping).
        """
        return (0 if self.outcome is None else self.outcome[self.player],
                self.explore_count, self.total_reward)

    def best_child(self):
        """Returns the best child in order of the sort key."""
        return max(self.children, key=SearchNode.sort_key)

    def children_str(self, state=None):
        """Returns the string representation of this node's children.

        They are ordered based on the sort key, so order of being chosen to play.

        Args:
        state: A `pyspiel.State` object, to be used to convert the action id into
            a human readable format. If None, the action integer id is used.
        """
        return "\n".join([
            c.to_str(state)
            for c in reversed(sorted(self.children, key=SearchNode.sort_key))
        ])

    def to_str(self, state=None):
        """Returns the string representation of this node.

        Args:
        state: A `pyspiel.State` object, to be used to convert the action id into
            a human readable format. If None, the action integer id is used.
        """
        action = (
            state.action_to_string(state.current_player(), self.action)
            if state and self.action is not None else str(self.action))
        return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
                "outcome: {}, reward: {:4.1f}, {:3d} children").format(
                    action, self.player, self.prior, self.explore_count and
                    self.total_reward / self.explore_count, self.explore_count,
                    ("{:4.1f}".format(self.outcome[self.player])
                    if self.outcome else "none"), self.total_reward, len(self.children))

    def __str__(self):
        return self.to_str(None)

class CustomBot(pyspiel.Bot):
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(self,
                game,
                uct_c,
                max_simulations,
                evaluator,
                player_id,
                child_selection_fn=SearchNode.puct_value, # SearchNode.uct_value,
                dirichlet_noise=None):
        """Initializes a MCTS Search algorithm in the form of a bot.

        uct_c: The exploration constant for UCT.
        max_simulations: How many iterations of MCTS to perform. Each simulation
            will result in one call to the evaluator. Memory usage should grow
            linearly with simulations * branching factor.
        child_selection_fn: A function to select the child in the descent phase.
            The default is UCT.
        dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
            the policy at the root. This is from the alpha-zero paper.

        ValueError if game.get_type().reward_model != pyspiel.GameType.RewardModel.TERMINAL and game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL
        """
        pyspiel.Bot.__init__(self)

        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.player_id = player_id
        self.solve = True # Whether to back up solved states.
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = None or np.random.RandomState() # An optional numpy RandomState to make it deterministic.
        self._child_selection_fn = child_selection_fn
        self.information = []

    def __str__(self):
        return "custom"

    def init_knowledge(self):
        nbr_piece_left = np.array([1, 6, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1])
        moved_before = np.zeros((10, 10))
        moved_scout = np.zeros((10, 10))
        matrix_of_stat = matrix_of_stats(self.player_id)
        self.information = [self.player_id, nbr_piece_left, moved_before, moved_scout, matrix_of_stat]
        self.last_moves = [None, None, None, None, None]
        self.evaluator.set_information(self.information, self.last_moves) # Make a link so that the evaluator has access to the information

    def update_knowledge(self, state, action):
        updating_knowledge(self.information, state, action)

    def step(self, state):
        root = self.mcts_search(state)
        log(root.children_str(state))
        mcts_action = root.best_child().action
        self.last_moves[1:] = self.last_moves[:-1]
        self.last_moves[0] = mcts_action
        return mcts_action

    def _apply_tree_policy(self, root, state):
        """Applies the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.

        Args:
        root: The root node in the search tree.
        state: The state of the game at the root node.

        Returns:
        visit_path: A list of nodes descending from the root node to a leaf node.
        working_state: The state of the game at the leaf node.
        """
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while (not working_state.is_terminal() and current_node.explore_count > 0):
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_state, 0)
                # # Since dirichlet_noise is set to None, this will never be true.
                # if current_node is root and self._dirichlet_noise:
                #     print("test")
                #     epsilon, alpha = self._dirichlet_noise
                #     noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                #     legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                #                 for (a, p), n in zip(legal_actions, noise)]

                # Reduce bias from move generation order
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                    SearchNode(action, player, prior) for action, prior in legal_actions
                ]

            # Choose node with largest UCT value
            chosen_child = max(current_node.children,
                    key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                        c, current_node.explore_count, self.uct_c))

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    def mcts_search(self, state):
        """A vanilla Monte-Carlo Tree Search algorithm.

        From state, search tree and at leaf, evaluator is called (if not terminal)
        A total of max_simulations states are explored.

        At every node, the algorithm chooses the action with the highest PUCT value,
        defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
        reward after the action, and N is the number of times the action was
        explored in this position. The input parameter c controls the balance
        between exploration and exploitation; higher values of c encourage
        exploration of under-explored nodes. Unseen actions are always explored
        first.

        At the end of the search, the chosen action is the action that has been
        explored most often. This is the action that is returned.

        Returns: The most visited move from the root node.
        """
        root = SearchNode(None, state.current_player(), 0)
        for _ in range(self.max_simulations):
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False

            while visit_path:
                # For chance nodes, walk up the tree to find the decision-maker.
                decision_node_idx = -1
                while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
                    decision_node_idx -= 1
                # Chance node targets are for the respective decision-maker.
                target_return = returns[visit_path[decision_node_idx].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    # If any have max utility (won?), or all children are solved,
                    # choose the one best for the player choosing.
                    best = None
                    all_solved = True
                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif best is None or child.outcome[player] > best.outcome[player]:
                            best = child
                    if (best is not None and
                        (all_solved or best.outcome[player] == self.max_utility)):
                        node.outcome = best.outcome
                    else:
                        solved = False
            if root.outcome is not None:
                break
        return root
    