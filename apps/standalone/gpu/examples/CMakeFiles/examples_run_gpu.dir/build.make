# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron

# Include any dependencies generated for this target.
include apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/depend.make

# Include the progress variables for this target.
include apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/flags.make

apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o: apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/flags.make
apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o: apps/standalone/gpu/examples/examplemain.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o -c /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples/examplemain.cpp

apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/examples_run_gpu.dir/examplemain.cpp.i"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples/examplemain.cpp > CMakeFiles/examples_run_gpu.dir/examplemain.cpp.i

apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/examples_run_gpu.dir/examplemain.cpp.s"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples/examplemain.cpp -o CMakeFiles/examples_run_gpu.dir/examplemain.cpp.s

# Object files for target examples_run_gpu
examples_run_gpu_OBJECTS = \
"CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o"

# External object files for target examples_run_gpu
examples_run_gpu_EXTERNAL_OBJECTS =

apps/standalone/gpu/examples/examples_run_gpu: apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/examplemain.cpp.o
apps/standalone/gpu/examples/examples_run_gpu: apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/build.make
apps/standalone/gpu/examples/examples_run_gpu: apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable examples_run_gpu"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/examples_run_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/build: apps/standalone/gpu/examples/examples_run_gpu

.PHONY : apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/build

apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/clean:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples && $(CMAKE_COMMAND) -P CMakeFiles/examples_run_gpu.dir/cmake_clean.cmake
.PHONY : apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/clean

apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/depend:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/standalone/gpu/examples/CMakeFiles/examples_run_gpu.dir/depend

