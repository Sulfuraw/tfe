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
include bots/human/CMakeFiles/human_bot.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bots/human/CMakeFiles/human_bot.dir/compiler_depend.make

# Include the progress variables for this target.
include bots/human/CMakeFiles/human_bot.dir/progress.make

# Include the compile flags for this target's objects.
include bots/human/CMakeFiles/human_bot.dir/flags.make

bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o: bots/human/CMakeFiles/human_bot.dir/flags.make
bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o: /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/human/human_bot.cc
bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o: bots/human/CMakeFiles/human_bot.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomas/Bureau/tfe/open_spiel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots/human && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o -MF CMakeFiles/human_bot.dir/human_bot.cc.o.d -o CMakeFiles/human_bot.dir/human_bot.cc.o -c /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/human/human_bot.cc

bots/human/CMakeFiles/human_bot.dir/human_bot.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/human_bot.dir/human_bot.cc.i"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots/human && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/human/human_bot.cc > CMakeFiles/human_bot.dir/human_bot.cc.i

bots/human/CMakeFiles/human_bot.dir/human_bot.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/human_bot.dir/human_bot.cc.s"
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots/human && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/human/human_bot.cc -o CMakeFiles/human_bot.dir/human_bot.cc.s

human_bot: bots/human/CMakeFiles/human_bot.dir/human_bot.cc.o
human_bot: bots/human/CMakeFiles/human_bot.dir/build.make
.PHONY : human_bot

# Rule to build all files generated by this target.
bots/human/CMakeFiles/human_bot.dir/build: human_bot
.PHONY : bots/human/CMakeFiles/human_bot.dir/build

bots/human/CMakeFiles/human_bot.dir/clean:
	cd /home/thomas/Bureau/tfe/open_spiel/build/bots/human && $(CMAKE_COMMAND) -P CMakeFiles/human_bot.dir/cmake_clean.cmake
.PHONY : bots/human/CMakeFiles/human_bot.dir/clean

bots/human/CMakeFiles/human_bot.dir/depend:
	cd /home/thomas/Bureau/tfe/open_spiel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomas/Bureau/tfe/open_spiel/open_spiel /home/thomas/Bureau/tfe/open_spiel/open_spiel/bots/human /home/thomas/Bureau/tfe/open_spiel/build /home/thomas/Bureau/tfe/open_spiel/build/bots/human /home/thomas/Bureau/tfe/open_spiel/build/bots/human/CMakeFiles/human_bot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bots/human/CMakeFiles/human_bot.dir/depend
