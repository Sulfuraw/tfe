# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thomas/Bureau/tfe/open_spiel/open_spiel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomas/Bureau/tfe/open_spiel/build

# Include any dependencies generated for this target.
include bots/CMakeFiles/bots.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bots/CMakeFiles/bots.dir/compiler_depend.make

# Include the progress variables for this target.
include bots/CMakeFiles/bots.dir/progress.make

# Include the compile flags for this target's objects.
include bots/CMakeFiles/bots.dir/flags.make

bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o: bots/CMakeFiles/bots.dir/flags.make
bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o: /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/gin_rummy/simple_gin_rummy_bot.cc
bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o: bots/CMakeFiles/bots.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomas/Bureau/tfe/open_spiel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o -MF CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o.d -o CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o -c /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/gin_rummy/simple_gin_rummy_bot.cc

bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.i"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/gin_rummy/simple_gin_rummy_bot.cc > CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.i

bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.s"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/gin_rummy/simple_gin_rummy_bot.cc -o CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.s

bots: bots/CMakeFiles/bots.dir/gin_rummy/simple_gin_rummy_bot.cc.o
bots: bots/CMakeFiles/bots.dir/build.make
.PHONY : bots

# Rule to build all files generated by this target.
bots/CMakeFiles/bots.dir/build: bots
.PHONY : bots/CMakeFiles/bots.dir/build

bots/CMakeFiles/bots.dir/clean:
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots && $(CMAKE_COMMAND) -P CMakeFiles/bots.dir/cmake_clean.cmake
.PHONY : bots/CMakeFiles/bots.dir/clean

bots/CMakeFiles/bots.dir/depend:
	cd /home/thomas/Bureau/tfe/open_spiel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomas/Bureau/tfe/open_spiel/open_spiel /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots /home/thomas/Bureau/tfe/open_spiel/build /home/thomas/Bureau/tfe/open_spiel/build/bots /home/thomas/Bureau/tfe/open_spiel/build/bots/CMakeFiles/bots.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bots/CMakeFiles/bots.dir/depend

