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
include apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/depend.make

# Include the progress variables for this target.
include apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/progress.make

# Include the compile flags for this target's objects.
include apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/flags.make

apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o: apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/flags.make
apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o: apps/standalone/gpu/deblurring/2d/deblur_2d_cg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o -c /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d/deblur_2d_cg.cpp

apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.i"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d/deblur_2d_cg.cpp > CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.i

apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.s"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d/deblur_2d_cg.cpp -o CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.s

# Object files for target deblur_2d_cg
deblur_2d_cg_OBJECTS = \
"CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o"

# External object files for target deblur_2d_cg
deblur_2d_cg_EXTERNAL_OBJECTS =

apps/standalone/gpu/deblurring/2d/deblur_2d_cg: apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/deblur_2d_cg.cpp.o
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/build.make
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/core/cpu/hostutils/libgadgetron_toolbox_hostutils.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/operators/gpu/libgadgetron_toolbox_gpuoperators.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/solvers/gpu/libgadgetron_toolbox_gpusolvers.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/local/cuda-10.0/lib64/libcudart.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/nfft/gpu/libgadgetron_toolbox_gpunfft.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/fft/gpu/libgadgetron_toolbox_gpufft.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/core/gpu/libgadgetron_toolbox_gpucore.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/core/cpu/libgadgetron_toolbox_cpucore.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_system.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_timer.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib/libismrmrd.so.1.4.2
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libpthread.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libsz.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libz.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libdl.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libm.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/lib/x86_64-linux-gnu/libboost_python3.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/local/cuda-10.0/lib64/libcufft.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/local/cuda-10.0/lib64/libcublas.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/local/cuda-10.0/lib64/libcusparse.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: /usr/local/cuda-10.0/lib64/libcudart.so
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: toolboxes/log/libgadgetron_toolbox_log.so.4.1.1
apps/standalone/gpu/deblurring/2d/deblur_2d_cg: apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable deblur_2d_cg"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deblur_2d_cg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/build: apps/standalone/gpu/deblurring/2d/deblur_2d_cg

.PHONY : apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/build

apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/clean:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d && $(CMAKE_COMMAND) -P CMakeFiles/deblur_2d_cg.dir/cmake_clean.cmake
.PHONY : apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/clean

apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/depend:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/standalone/gpu/deblurring/2d/CMakeFiles/deblur_2d_cg.dir/depend
