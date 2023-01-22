# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that Python and C++ bots can be called by a C++ algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from open_spiel.python.bots import uniform_random
import pyspiel

# Specify bot names in alphabetical order, to make it easier to read.
SPIEL_BOTS_LIST = [
    "fixed_action_preference",
    "uniform_random",
]


class BotTest(absltest.TestCase):

  def test_python_and_cpp_bot(self):
    game = pyspiel.load_game("kuhn_poker")
    bots = [
        pyspiel.make_uniform_random_bot(0, 1234),
        uniform_random.UniformRandomBot(1, np.random.RandomState(4321)),
    ]
    results = np.array([
        pyspiel.evaluate_bots(game.new_initial_state(), bots, iteration)
        for iteration in range(10000)
    ])
    average_results = np.mean(results, axis=0)
    np.testing.assert_allclose(average_results, [0.125, -0.125], atol=0.1)

  def test_registered_bots(self):
    self.assertCountEqual(pyspiel.registered_bots(), SPIEL_BOTS_LIST)

  def test_cpp_mcts_bot(self):
    game = pyspiel.load_game("tic_tac_toe")
    bots = [
        pyspiel.MCTSBot(game, pyspiel.RandomRolloutEvaluator(1, 0), 2.0,
                        100, 100, False, 42, False)
    ] * 2
    _ = np.array([
        pyspiel.evaluate_bots(game.new_initial_state(), bots, iteration)
        for iteration in range(10)
    ])
    # Do a search directly, and inspect the values.
    state = game.new_initial_state()
    search_node = bots[0].mcts_search(state)
    for child in search_node.children:
      print(f"Child action {child.action}, total reward: {child.total_reward}" +
            f", explore count: {child.explore_count}")
    # Similar way to achieve the above.
    print(f"Children string: {search_node.children_str(state)}")
    print(f"Best child: {search_node.best_child().to_string(state)}")

  def test_can_play_game(self):
    game = pyspiel.load_game("kuhn_poker")
    self.assertIn("uniform_random", pyspiel.bots_that_can_play_game(game))

  def test_passing_params(self):
    game = pyspiel.load_game("tic_tac_toe")
    bots = [
        pyspiel.load_bot(
            "fixed_action_preference",
            game,
            player=0,
            params={"actions": "0:1:2"}),
        pyspiel.load_bot(
            "fixed_action_preference",
            game,
            player=1,
            params={"actions": "3:4"}),
    ]
    result = pyspiel.evaluate_bots(game.new_initial_state(), bots, seed=0)
    self.assertEqual(result, [1, -1])  # Player 0 wins.


if __name__ == "__main__":
  absltest.main()
