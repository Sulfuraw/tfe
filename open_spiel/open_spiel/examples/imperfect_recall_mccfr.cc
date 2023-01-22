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

#include <ostream>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/games/phantom_ttt.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// E.g. another choice: dark_hex_ir(board_size=2)
ABSL_FLAG(std::string, game, "phantom_ttt_ir", "Game string");
ABSL_FLAG(int, num_iters, 1000000, "How many iters to run for.");
ABSL_FLAG(int, report_every, 1000, "How often to report.");

namespace open_spiel {
namespace {

void ImperfectRecallMCCFR() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  // algorithms::ExternalSamplingMCCFRSolver solver(*game);
  algorithms::OutcomeSamplingMCCFRSolver solver(*game);

  for (int i = 0; i < absl::GetFlag(FLAGS_num_iters); ++i) {
    solver.RunIteration();

    // Do not run exploitability or NashConv, as it does not support imperfect
    // recall games. Simply print out the average strategy.
    if (i % absl::GetFlag(FLAGS_report_every) == 0 ||
        i == absl::GetFlag(FLAGS_num_iters) - 1) {
      std::cerr << "Iteration " << i << " average policy is " << std::endl;
      for (const auto& key_and_values : solver.InfoStateValuesTable()) {
        std::cout << "infostate key: " << std::endl
                  << key_and_values.first << std::endl
                  << "values: " << std::endl
                  << key_and_values.second.ToString() << std::endl;
      }
    }
  }
}
}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  open_spiel::ImperfectRecallMCCFR();
}
