# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin

# Include any dependencies generated for this target.
include examples/seir/CMakeFiles/quality.dir/depend.make

# Include the progress variables for this target.
include examples/seir/CMakeFiles/quality.dir/progress.make

# Include the compile flags for this target's objects.
include examples/seir/CMakeFiles/quality.dir/flags.make

examples/seir/CMakeFiles/quality.dir/quality.cpp.o: examples/seir/CMakeFiles/quality.dir/flags.make
examples/seir/CMakeFiles/quality.dir/quality.cpp.o: ../examples/seir/quality.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/seir/CMakeFiles/quality.dir/quality.cpp.o"
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/quality.dir/quality.cpp.o -c /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/examples/seir/quality.cpp

examples/seir/CMakeFiles/quality.dir/quality.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quality.dir/quality.cpp.i"
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/examples/seir/quality.cpp > CMakeFiles/quality.dir/quality.cpp.i

examples/seir/CMakeFiles/quality.dir/quality.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quality.dir/quality.cpp.s"
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/examples/seir/quality.cpp -o CMakeFiles/quality.dir/quality.cpp.s

# Object files for target quality
quality_OBJECTS = \
"CMakeFiles/quality.dir/quality.cpp.o"

# External object files for target quality
quality_EXTERNAL_OBJECTS =

examples/seir/quality: examples/seir/CMakeFiles/quality.dir/quality.cpp.o
examples/seir/quality: examples/seir/CMakeFiles/quality.dir/build.make
examples/seir/quality: libdnn_opt_core.a
examples/seir/quality: examples/seir/CMakeFiles/quality.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable quality"
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quality.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/seir/CMakeFiles/quality.dir/build: examples/seir/quality

.PHONY : examples/seir/CMakeFiles/quality.dir/build

examples/seir/CMakeFiles/quality.dir/clean:
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir && $(CMAKE_COMMAND) -P CMakeFiles/quality.dir/cmake_clean.cmake
.PHONY : examples/seir/CMakeFiles/quality.dir/clean

examples/seir/CMakeFiles/quality.dir/depend:
	cd /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/examples/seir /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir /home/jrojasdelgado/Documents/github/cys-covid/dnn_opt_seir/bin/examples/seir/CMakeFiles/quality.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/seir/CMakeFiles/quality.dir/depend

