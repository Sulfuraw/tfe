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

#ifndef OPEN_SPIEL_ALGORITHMS_NFG_WRITER_H_
#define OPEN_SPIEL_ALGORITHMS_NFG_WRITER_H_

#include <string>

#include "open_spiel/spiel.h"

namespace open_spiel {

// Functions to export normal-form games to Gambit's .nfg format.
// http://www.gambit-project.org/gambit13/formats.html#the-strategic-game-nfg-file-format-payoff-version

// Get the string representation of this normal-form game.
const std::string GameToNFGString(const Game& game);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_NFG_WRITER_H_
