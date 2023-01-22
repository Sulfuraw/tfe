# CMake generated Testfile for 
# Source directory: /home/thomas/Bureau/tfe/open_spiel/open_spiel/examples
# Build directory: /home/thomas/Bureau/tfe/open_spiel/build/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(benchmark_game_test "benchmark_game" "--game=tic_tac_toe" "--sims=100" "--attempts=2")
set_tests_properties(benchmark_game_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;2;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(cfr_example_test "cfr_example")
set_tests_properties(cfr_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;5;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(example_test "example" "--game=tic_tac_toe" "--seed=0")
set_tests_properties(example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;14;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(matrix_example_test "matrix_example")
set_tests_properties(matrix_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;19;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(mcts_example_test "mcts_example")
set_tests_properties(mcts_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;22;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(minimax_example_test "minimax_example")
set_tests_properties(minimax_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;25;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(policy_iteration_example_test "policy_iteration_example")
set_tests_properties(policy_iteration_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;28;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
add_test(value_iteration_example_test "value_iteration_example")
set_tests_properties(value_iteration_example_test PROPERTIES  _BACKTRACE_TRIPLES "/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;31;add_test;/home/thomas/Bureau/tfe/open_spiel/open_spiel/examples/CMakeLists.txt;0;")
