from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyspiel
import numpy as np
import math
from statework import *

# Using the Copyright 2019 DeepMind Technologies Limited using modified version of """Monte-Carlo Tree Search algorithm for game play"""

class RandomRolloutEvaluator():
    """A simple evaluator doing random rollouts.

    This evaluator returns the average outcome of playing random actions from the
    given state until the end of the game.  n_rollouts is the number of random
    outcomes to be considered.
    """
    def __init__(self, n_rollouts=5, n_move_before=10, random_state=None):
        self.n_rollouts = n_rollouts
        self.n_moves_before = n_move_before
        self._random_state = random_state or np.random.RandomState()

    def evaluate_state(self, state):
        """Returns evaluation on given state."""
        state_str = str(state)[:100].upper()
        nbr_pieces = [[0]*12, [0]*12]
        advanced_pieces = [0, 0]
        advanced_miner = [0, 0]
        place = 0
        for piece in state_str:
            if piece not in ["A", "_"]:
                i, player = pieces_to_index()[piece]
                nbr_pieces[player][i] += 1
                if player and place < 40:
                    advanced_pieces[player] += 1
                    if i == 4:
                        advanced_miner[player] += 1
                if not player and place > 60:
                    advanced_pieces[player] += 1
                    if i == 4:
                        advanced_miner[player] += 1
            place += 1

        score = [0, 0]
        for player in [0, 1]:
            for piece_id in range(12):
                score[player] += nbr_pieces[player][piece_id]
            # More weight having piece on the other side
            score[player] += advanced_pieces[player] * 0.02

        returns = [0, 0]
        for player in [0, 1]:
            x = score[player] - score[1-player]
            returns[player] = 2*(x - (-40))/(40 - (-40)) - 1.0 

        if returns[0] > 1 or returns[0] < -1 or returns[1] > 1 or returns[1] < -1:
            # Check if we need to adapt the rescale above
            print("==============================")
            print("Returns:", returns, "\n")
            printCharMatrix(str(state))
        return returns

    def evaluate(self, state):
        """Returns simulations / rollout of a state."""
        result = None
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            i = 0
            while not working_state.is_terminal() and i < self.n_moves_before:
                action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
                i += 1
            returns = np.array(working_state.returns()) if working_state.is_terminal() else np.array(self.evaluate_state(working_state))
            result = returns if result is None else result + returns
        return result / self.n_rollouts

    def prior(self, state):
        """Returns equal probability for all actions."""
        legal_actions = state.legal_actions(state.current_player())
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

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
        # print("win/game, right part:", ((self.total_reward / self.explore_count)/2 + 1/2), uct_c*math.sqrt(math.log(parent_explore_count)/self.explore_count))
        return (self.total_reward / self.explore_count)/2 + 1/2 + uct_c * math.sqrt(math.log(parent_explore_count) / self.explore_count)

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


class mctsBot(pyspiel.Bot):
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
        self.last_moves = [None, None, None, None, None]

    def restart_at(self, state):
        pass

    def __str__(self):
        return "mcts"

    def set_max_simulations(self, new_value):
        self.max_simulations = new_value

    def init_knowledge(self):
        nbr_piece_left = np.array([1, 6, 1, 8, 5, 4, 4, 4, 3, 2, 1, 1])
        moved_before = np.zeros((10, 10))
        moved_scout = np.zeros((10, 10))
        matrix_of_stat = matrix_of_stats(self.player_id)
        self.information = [self.player_id, nbr_piece_left, moved_before, moved_scout, matrix_of_stat]

    def update_knowledge(self, state, action):
        updating_knowledge(self.information, state, action)

    def step(self, state):
        root = self.mcts_search(state)
        mcts_action = root.best_child().action
        self.last_moves = [mcts_action, self.last_moves[0], self.last_moves[1], self.last_moves[2], self.last_moves[3]]
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
                
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                    SearchNode(action, player, prior) for action, prior in legal_actions
                ]

            # Otherwise (not chance node) choose node with largest UCT value
            chosen_child = max(current_node.children,
                    key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                        c, current_node.explore_count, self.uct_c))

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    def mcts_search(self, state):
        """A vanilla Monte-Carlo Tree Search algorithm.

        This algorithm searches the game tree from the given state.
        At the leaf, the evaluator is called if the game state is not terminal.
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

        This implementation supports sequential n-player games, with or without
        chance nodes. All players maximize their own reward and ignore the other
        players' rewards. This corresponds to max^n for n-player games. It is the
        norm for zero-sum games, but doesn't have any special handling for
        non-zero-sum games. It doesn't have any special handling for imperfect
        information games.

        The implementation also supports backing up solved states, i.e. MCTS-Solver.
        The implementation is general in that it is based on a max^n backup (each
        player greedily chooses their maximum among proven children values, or there
        exists one child whose proven value is game.max_utility()), so it will work
        for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
        draw games). Also chance nodes are considered proven only if all children
        have the same value.

        Arguments:
        state: pyspiel.State object, state to search from

        Returns:
        The most visited move from the root node.
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
    