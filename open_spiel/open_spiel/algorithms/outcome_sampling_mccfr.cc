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

#include "open_spiel/algorithms/outcome_sampling_mccfr.h"

#include <cmath>
#include <numeric>
#include <random>

#include "open_spiel/abseil-cpp/absl/random/discrete_distribution.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// Serialization of the MCCFR solver is in agreement with de/serialization of
// regular CFR solvers, i.e. take a look at the PartiallyDeserializedCFRSolver()
// method for more info.
constexpr const int kSerializationVersion = 1;
constexpr const char* kSerializeSolverRNGSectionHeader = "[SolverRNG]";
constexpr const char* kSerializeSolverEpsilonSectionHeader = "[SolverEpsilon]";
constexpr const char* kSerializeSolverDefaultPolicySectionHeader =
    "[SolverDefaultPolicy]";

OutcomeSamplingMCCFRSolver::OutcomeSamplingMCCFRSolver(const Game& game,
                                                       double epsilon, int seed)
    : OutcomeSamplingMCCFRSolver(game, std::make_shared<UniformPolicy>(),
                                 epsilon, seed) {}

OutcomeSamplingMCCFRSolver::OutcomeSamplingMCCFRSolver(
    const Game& game, std::shared_ptr<Policy> default_policy, double epsilon,
    int seed)
    : OutcomeSamplingMCCFRSolver(
          game.shared_from_this(), default_policy, epsilon,
          std::mt19937(seed >= 0 ? seed : std::mt19937::default_seed)) {}

OutcomeSamplingMCCFRSolver::OutcomeSamplingMCCFRSolver(
    std::shared_ptr<const Game> game, std::shared_ptr<Policy> default_policy,
    double epsilon, std::mt19937 rng)
    : game_(game),
      epsilon_(epsilon),
      rng_(rng),
      dist_(0.0, 1.0),
      default_policy_(default_policy) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}

void OutcomeSamplingMCCFRSolver::RunIteration(std::mt19937* rng) {
  // for (Player update_player = Player{0}; update_player < game_->NumPlayers();
  // ++update_player) {
  for (Player p = Player{0}; p < game_->NumPlayers(); ++p) {
    std::unique_ptr<State> state = game_->NewInitialState();
    SampleEpisode(state.get(), p, rng, 1.0, 1.0, 1.0);
  }
}

std::string OutcomeSamplingMCCFRSolver::Serialize(int double_precision,
                                                  std::string delimiter) const {
  SPIEL_CHECK_GE(double_precision, -1);
  std::string str = "";
  // Meta section
  absl::StrAppend(&str,
                  "# Automatically generated by OpenSpiel "
                  "OutcomeSamplingMCCFRSolver::Serialize\n");
  absl::StrAppend(&str, kSerializeMetaSectionHeader, "\n");
  absl::StrAppend(&str, "Version: ", kSerializationVersion, "\n");
  absl::StrAppend(&str, "\n");
  // Game section
  absl::StrAppend(&str, kSerializeGameSectionHeader, "\n");
  absl::StrAppend(&str, game_->Serialize(), "\n");
  // Internal solver state section
  absl::StrAppend(&str, kSerializeSolverTypeSectionHeader, "\n");
  absl::StrAppend(&str, "OutcomeSamplingMCCFRSolver", "\n");
  absl::StrAppend(&str, kSerializeSolverSpecificStateSectionHeader, "\n");
  // RNG section
  absl::StrAppend(&str, kSerializeSolverRNGSectionHeader, "\n");
  std::ostringstream rng_stream;
  rng_stream << rng_;
  absl::StrAppend(&str, rng_stream.str(), "\n");
  // Epsilon section
  absl::StrAppend(&str, kSerializeSolverEpsilonSectionHeader, "\n");
  absl::StrAppend(&str, epsilon_, "\n");
  // Default policy section
  absl::StrAppend(&str, kSerializeSolverDefaultPolicySectionHeader, "\n");
  absl::StrAppend(&str, default_policy_->Serialize(double_precision, delimiter),
                  "\n");
  // Info state values table section
  absl::StrAppend(&str, kSerializeSolverValuesTableSectionHeader, "\n");
  SerializeCFRInfoStateValuesTable(info_states_, &str, double_precision,
                                   delimiter);
  return str;
}

std::vector<double> OutcomeSamplingMCCFRSolver::SamplePolicy(
    const CFRInfoStateValues& info_state) const {
  std::vector<double> policy = info_state.current_policy;
  for (int i = 0; i < policy.size(); ++i) {
    policy[i] = epsilon_ * 1.0 / policy.size() + (1 - epsilon_) * policy[i];
  }
  return policy;
}

double OutcomeSamplingMCCFRSolver::Baseline(
    const State& state, const CFRInfoStateValues& info_state, int aidx) const {
  // Default to vanilla outcome sampling.
  return 0;
}

// Applies Eq. 9 of Schmid et al. '19
double OutcomeSamplingMCCFRSolver::BaselineCorrectedChildValue(
    const State& state, const CFRInfoStateValues& info_state, int sampled_aidx,
    int aidx, double child_value, double sample_prob) const {
  double baseline = Baseline(state, info_state, aidx);
  if (aidx == sampled_aidx) {
    return baseline + (child_value - baseline) / sample_prob;
  } else {
    return baseline;
  }
}

double OutcomeSamplingMCCFRSolver::SampleEpisode(
    State* state, Player update_player, std::mt19937* rng, double my_reach,
    double opp_reach, double sample_reach) {
  if (state->IsTerminal()) {
    return state->PlayerReturn(update_player);
  } else if (state->IsChanceNode()) {
    std::pair<Action, double> outcome_and_prob =
        SampleAction(state->ChanceOutcomes(), dist_(*rng));
    SPIEL_CHECK_PROB(outcome_and_prob.second);
    SPIEL_CHECK_GT(outcome_and_prob.second, 0);
    state->ApplyAction(outcome_and_prob.first);
    return SampleEpisode(state, update_player, rng, my_reach,
                         outcome_and_prob.second * opp_reach,
                         outcome_and_prob.second * sample_reach);
  } else if (state->IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  SPIEL_CHECK_PROB(sample_reach);

  int player = state->CurrentPlayer();
  std::string is_key = state->InformationStateString(player);
  std::vector<Action> legal_actions = state->LegalActions();

  // The insert here only inserts the default value if the key is not found,
  // otherwise returns the entry in the map.
  auto iter_and_result = info_states_.insert(
      {is_key, CFRInfoStateValues(legal_actions, kInitialTableValues)});

  CFRInfoStateValues info_state_copy = iter_and_result.first->second;
  info_state_copy.ApplyRegretMatching();

  const std::vector<double>& sample_policy =
      (player == update_player ? SamplePolicy(info_state_copy)
                               : info_state_copy.current_policy);

  absl::discrete_distribution<int> action_dist(sample_policy.begin(),
                                               sample_policy.end());
  int sampled_aidx = action_dist(*rng);
  SPIEL_CHECK_PROB(sample_policy[sampled_aidx]);
  SPIEL_CHECK_GT(sample_policy[sampled_aidx], 0);

  state->ApplyAction(legal_actions[sampled_aidx]);
  double child_value = SampleEpisode(
      state, update_player, rng,
      player == update_player
          ? my_reach * info_state_copy.current_policy[sampled_aidx]
          : my_reach,
      player == update_player
          ? opp_reach
          : opp_reach * info_state_copy.current_policy[sampled_aidx],
      sample_reach * sample_policy[sampled_aidx]);

  // Compute each of the child estimated values.
  std::vector<double> child_values(legal_actions.size(), 0);
  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    child_values[aidx] =
        BaselineCorrectedChildValue(*state, info_state_copy, sampled_aidx, aidx,
                                    child_value, sample_policy[aidx]);
  }

  // Compute the value of this history for this policy.
  double value_estimate = 0;
  for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
    value_estimate +=
        info_state_copy.current_policy[sampled_aidx] * child_values[aidx];
  }

  if (player == update_player) {
    // Now the regret and avg strategy updates.
    CFRInfoStateValues& info_state = info_states_[is_key];
    info_state.ApplyRegretMatching();

    // Estimate for the counterfactual value of the policy.
    double cf_value = value_estimate * opp_reach / sample_reach;

    // Update regrets.
    //
    // Note: different from Chapter 4 of Lanctot '13 thesis, the utilities
    // coming back from the recursion are already multiplied by the players'
    // tail reaches and divided by the sample tail reach. So when adding regrets
    // to the table, we need only multiply by the opponent reach and divide by
    // the sample reach to this point.
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      // Estimate for the counterfactual value of the policy replaced by always
      // choosing sampled_aidx at this information state.
      double cf_action_value = child_values[aidx] * opp_reach / sample_reach;
      info_state.cumulative_regrets[aidx] += (cf_action_value - cf_value);
    }

    // Update the average policy.
    for (int aidx = 0; aidx < legal_actions.size(); ++aidx) {
      double increment =
          my_reach * info_state.current_policy[aidx] / sample_reach;
      SPIEL_CHECK_FALSE(std::isnan(increment) || std::isinf(increment));
      info_state.cumulative_policy[aidx] += increment;
    }
  }

  return value_estimate;
}

std::unique_ptr<OutcomeSamplingMCCFRSolver>
DeserializeOutcomeSamplingMCCFRSolver(const std::string& serialized,
                                      std::string delimiter) {
  auto partial = PartiallyDeserializeCFRSolver(serialized);
  SPIEL_CHECK_EQ(partial.solver_type, "OutcomeSamplingMCCFRSolver");

  enum Section { kInvalid = -1, kRNG = 0, kEpsilon = 1, kDefaultPolicy = 2 };
  std::array<std::string, 3> section_strings = {"", "", ""};
  Section current_section = kInvalid;

  std::vector<absl::string_view> lines =
      absl::StrSplit(partial.solver_specific_state, '\n');
  for (int i = 0; i < lines.size(); i++) {
    if (lines[i] == kSerializeSolverRNGSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kInvalid);
      current_section = kRNG;
    } else if (lines[i] == kSerializeSolverEpsilonSectionHeader) {
      SPIEL_CHECK_EQ(current_section, kRNG);
      current_section = kEpsilon;
    } else if (lines[i] == kSerializeSolverDefaultPolicySectionHeader) {
      SPIEL_CHECK_EQ(current_section, kEpsilon);
      current_section = kDefaultPolicy;
    } else {
      SPIEL_CHECK_NE(current_section, kInvalid);
      absl::StrAppend(&section_strings[current_section], lines[i], "\n");
    }
  }
  // Remove trailing newline characters
  for (int i = 0; i < section_strings.size(); i++) {
    section_strings[i].erase(section_strings[i].length() - 1);
  }

  // Do the actual deserialization for all internal state values
  std::mt19937 rng = std::mt19937();
  std::istringstream rng_stream(section_strings[kRNG]);
  rng_stream >> rng;

  // First scalar is epsilon, second is update_player
  std::string epsilon = section_strings[kEpsilon];

  auto solver = std::make_unique<OutcomeSamplingMCCFRSolver>(
      partial.game, DeserializePolicy(section_strings[kDefaultPolicy]),
      std::stod(epsilon), rng);
  DeserializeCFRInfoStateValuesTable(partial.serialized_cfr_values_table,
                                     &solver->InfoStateValuesTable(),
                                     delimiter);
  return solver;
}

}  // namespace algorithms
}  // namespace open_spiel
