// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_EVALUATE_BOTS_H_
#define OPEN_SPIEL_ALGORITHMS_EVALUATE_BOTS_H_

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {

// Play a game once, to compare bot performance.
// Must supply one bot for each player in the game.
// `state.Restart()` will be called if the state is the root node, otherwise
// `RestartAt(state)` will be called.
std::vector<double> EvaluateBots(State* state, const std::vector<Bot*>& bots,
                                 int seed);

// As above, but creates a new state at the root of the game.
std::vector<double> EvaluateBots(const Game& game,
                                 const std::vector<Bot*>& bots, int seed);

// As above, but calculates a random seed.
std::vector<double> EvaluateBots(const Game& game,
                                 const std::vector<Bot*>& bots);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_EVALUATE_BOTS_H_
