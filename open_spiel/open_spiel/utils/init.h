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

#ifndef OPEN_SPIEL_UTILS_INIT_H_
#define OPEN_SPIEL_UTILS_INIT_H_

namespace open_spiel {

// A utility function useful for mixing internal and external use of OpenSpiel.
// Intended to be called early in a program's main. Currently only necessary
// in programs that mix internal and external use (e.g. utils/file.h).
void Init(const char* usage, int* argc, char*** argv, bool remove_flags);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_INIT_H_
