from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import numpy as np
import math
import time
from statework import *

# Using the Copyright 2019 DeepMind Technologies Limited using modified version of """Monte-Carlo Tree Search algorithm for game play"""
class CustomEvaluator():
    def __init__(self):
        self.piece_to_index = pieces_to_index()
        self.player_pieces = players_pieces()

    def set_information(self, information):
        self.information = information
    
    def value_for_piece(self, enemies):
        value = np.array([0.02]*12)
        for i in range(4, 12):
            value[i] = value[i-1]*1.6
        value[0] = 1.0
        value[1] = 0.9 if enemies[4] == 0 else 0.8
        value[2] = 0.8 if enemies[11] else 0.02
        value[4] += 0.032 if enemies[1] < 3 else 0.6
        returns = []
        for i in range(12):
            returns.append((i, value[i]))
        return returns

    def evaluate_state(self, state, move_of_state):
        """Returns evaluation on given state."""
        state_str = str(state)

        nbr_pieces = [[0]*12, [0]*12]
        for piece in state_str[:100].upper():
            if piece not in ["A", "_"]:
                i, player = self.piece_to_index[piece]
                nbr_pieces[player][i] += 1

        score = [0, 0]
        for player in [0, 1]:
            for piece_id, value in self.value_for_piece(nbr_pieces[1-player]):
                score[player] += nbr_pieces[player][piece_id]*value

                # make more weight having pieces on the other part of the board, go forward
                if player:
                    score[player] += state_str[:40].count(self.player_pieces[player][piece_id]) * 0.01 # (old) 1/40 = 0.025
                else:
                    score[player] += state_str[-40:].count(self.player_pieces[player][piece_id]) * 0.01 # Mtn c'est * 0.01 parce 0.02 est la value la plus basse d'une pièce

        returns = [0, 0]
        for player in [0, 1]:
            returns[player] = score[player] - score[1-player]   # Re-range with (x - min)/(max-min)
        # Print for debug
        # if returns[0] > 1 or returns[0] < -1 or returns[1] > 1 or returns[1] < -1:
        #     print("==============================")
        #     print("Here, the returns were strange:", returns, "\n")
        #     print(state_str)
        return returns    

    def evaluate(self, state):
        """Returns evaluation on given state."""
        result = None
        move_of_state = int(str(state)[103-len(str(state)):])
        
        if move_of_state < 50:
            n_rollouts = 8
            n_moves_before = 4
        if move_of_state < 100:
            n_rollouts = 8
            n_moves_before = 8
        elif move_of_state < 150:
            n_rollouts = 8
            n_moves_before = 10
        else:
            n_rollouts = 10
            n_moves_before = 20

        for _ in range(n_rollouts):
            working_state = state.clone()
            i = 0
            while not working_state.is_terminal() and i < n_moves_before:
                # We simulate our moves with prior and simulate ennemy move randomly.
                if True: # TODO: Change: (i%2) == 0
                    legal_actions = self.prior(working_state)
                    actions, proba = list(zip(*legal_actions))

                    # Transform proba to delete values below a treshold and help converge the results
                    proba = np.array(proba)
                    if np.max(proba) > 0.045: #(move_of_state+i) > 100
                        proba[proba <= 0.045] = 0
                        proba = proba/np.sum(proba)

                    action = np.random.choice(actions, p=proba)
                else:
                    action = np.random.choice(working_state.legal_actions())
                working_state.apply_action(action)
                i += 1
            # We sum the returns with terminal*5 if the last state is terminal, the evaluation of the state if it's not
            returns = np.array(working_state.returns())*40 if working_state.is_terminal() else np.array(self.evaluate_state(working_state, move_of_state+i))
            result = returns if result is None else result + returns
        return result / n_rollouts
    
    def toward_flag(self, state, player, coord):
        flag_str_pos = str(state).find(players_pieces()[1-player][0])
        flag = [flag_str_pos//10, flag_str_pos%10]
        man_distance_before = abs(flag[0] - coord[1]) + abs(flag[1] - coord[0])
        man_distance_after = abs(flag[0] - coord[3]) + abs(flag[1] - coord[2])
        # Categorize the weight
        value = 21-man_distance_after
        if value < 11: value = 5
        elif value < 16: value = 10
        else: value = 15
        return man_distance_before > man_distance_after, value
    
    def proba_win_combat(self, ally, enemy_pos, state, information):
        def win_combat(ally, enemy):
            """Only work on moveable ally at the moment"""
            allyIdx, _ = pieces_to_index()[ally]
            enemyIdx, _ = pieces_to_index()[enemy]
            if enemyIdx == 1:
                return 1 if allyIdx == 4 else -1
            if enemyIdx == 11:
                return 1 if (allyIdx == 2 or allyIdx == 11) else -1
            return 1 if enemyIdx <= allyIdx else -1
        _, pieces_left, moved, scout, matrix_of_stats = information
        player = state.current_player()
        enemy_pieces = players_pieces()[1-player]
        state_str = str(state).upper()
        if sum(matrix_of_stats[enemy_pos[0]][enemy_pos[1]]) < 0.2:
            return win_combat(ally, state_str[enemy_pos[0]*10+enemy_pos[1]])
        elif scout[enemy_pos[0]][enemy_pos[1]]:
            return win_combat(ally, enemy_pieces[3])
        elif moved[enemy_pos[0]][enemy_pos[1]]:
            probas = moved_piece_matrix(pieces_left, matrix_of_stats[enemy_pos[0]][enemy_pos[1]])
        else:
            probas = no_info_piece_matrix(pieces_left, matrix_of_stats[enemy_pos[0]][enemy_pos[1]])
        summ = 0
        for idx in range(12):
            summ += probas[idx]*win_combat(ally, enemy_pieces[idx])
        return summ

    def prior(self, state):
        """Returns probability for each actions"""
        sum = 0.0
        prio = []
        player = state.current_player()
        state_str = str(state).upper()
        for action in state.legal_actions(player):
            coord = action_to_coord(state.action_to_string(player, action))
            # # Prioritize winning attacks / penalize loosing ones
            arrival = state_str[coord[3]*10 + coord[2]]
            # Fight
            if arrival != "A":
                ally = state_str[coord[1]*10 + coord[0]]
                value = self.proba_win_combat(ally, [coord[3], coord[2]], state, self.information)*20
                if value < 0: value = 1.0
            else:
                # Priorize moving toward the ennemy flag
                toward_flag = self.toward_flag(state, player, coord)
                value = toward_flag[1] if toward_flag[0] else 1.0
            sum += value
            prio.append([action, value])
        # Rescale to make the sum equal to 1
        for i in range(len(prio)):
            prio[i][1] = prio[i][1]/sum
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

        return self.total_reward / self.explore_count + uct_c * math.sqrt(
            math.log(parent_explore_count) / self.explore_count)

    def puct_value(self, parent_explore_count, uct_c):
        """Returns the PUCT value of child."""
        if self.outcome is not None:
            return self.outcome[self.player]

        return ((self.explore_count and self.total_reward / self.explore_count) +
                uct_c * self.prior * math.sqrt(parent_explore_count) /
                (self.explore_count + 1))

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
                "outcome: {}, {:3d} children").format(
                    action, self.player, self.prior, self.explore_count and
                    self.total_reward / self.explore_count, self.explore_count,
                    ("{:4.1f}".format(self.outcome[self.player])
                    if self.outcome else "none"), len(self.children))

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
                child_selection_fn=SearchNode.uct_value,
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

        self._game = game
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.player_id = player_id
        self.verbose = False # Whether to print information about the search tree before returning the action
        self.solve = True # Whether to back up solved states.
        self.max_utility = game.max_utility()
        self._dirichlet_noise = dirichlet_noise
        self._random_state = None or np.random.RandomState() # An optional numpy RandomState to make it deterministic.
        self._child_selection_fn = child_selection_fn

        self.information = []

    def __str__(self):
        return "custom"

    def set_max_simulations(self, new_value):
        self.max_simulations = new_value

    def init_knowledge(self):
        nbr_piece_left = np.array([1, 6, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1])
        moved_before = np.zeros((10, 10))
        moved_scout = np.zeros((10, 10))
        matrix_of_stat = matrix_of_stats(self.player_id)
        self.information = [self.player_id, nbr_piece_left, moved_before, moved_scout, matrix_of_stat]
        self.evaluator.set_information(self.information)

    def update_knowledge(self, state, action):
        updating_knowledge(self.information, state, action)

    def step(self, state):
        t1 = time.time()
        root = self.mcts_search(state)

        best = root.best_child()

        mcts_action = best.action

        # policy = [(action, (1.0 if action == mcts_action else 0.0))
        #         for action in state.legal_actions(state.current_player())]
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
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                                for (a, p), n in zip(legal_actions, noise)]
                # Reduce bias from move generation order
                # TODO: Verify that we can disable this because we have a prior: Chelou mais ça a vraiment l'air de rien changer
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
    