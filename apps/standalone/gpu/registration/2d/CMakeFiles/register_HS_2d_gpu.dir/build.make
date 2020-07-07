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
include apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/depend.make

# Include the progress variables for this target.
include apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/flags.make

apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o: apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/flags.make
apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o: apps/standalone/gpu/registration/2d/register_HS_2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o -c /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d/register_HS_2d.cpp

apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.i"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d/register_HS_2d.cpp > CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.i

apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.s"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d/register_HS_2d.cpp -o CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.s

# Object files for target register_HS_2d_gpu
register_HS_2d_gpu_OBJECTS = \
"CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o"

# External object files for target register_HS_2d_gpu
register_HS_2d_gpu_EXTERNAL_OBJECTS =

apps/standalone/gpu/registration/2d/register_HS_2d_gpu: apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/register_HS_2d.cpp.o
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/build.make
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/core/cpu/hostutils/libgadgetron_toolbox_hostutils.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/registration/optical_flow/gpu/libgadgetron_toolbox_gpureg.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/operators/gpu/libgadgetron_toolbox_gpuoperators.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/solvers/gpu/libgadgetron_toolbox_gpusolvers.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/local/cuda-10.0/lib64/libcudart.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/core/cpu/math/libgadgetron_toolbox_cpucore_math.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/liblapacke.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libopenblas.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/nfft/gpu/libgadgetron_toolbox_gpunfft.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/fft/gpu/libgadgetron_toolbox_gpufft.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/core/gpu/libgadgetron_toolbox_gpucore.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/local/cuda-10.0/lib64/libcufft.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/local/cuda-10.0/lib64/libcublas.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/core/cpu/libgadgetron_toolbox_cpucore.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_system.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_timer.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib/libismrmrd.so.1.4.2
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libpthread.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libsz.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libz.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libdl.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libm.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/lib/x86_64-linux-gnu/libboost_python3.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/local/cuda-10.0/lib64/libcusparse.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: /usr/local/cuda-10.0/lib64/libcudart.so
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: toolboxes/log/libgadgetron_toolbox_log.so.4.1.1
apps/standalone/gpu/registration/2d/register_HS_2d_gpu: apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable register_HS_2d_gpu"
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/register_HS_2d_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/build: apps/standalone/gpu/registration/2d/register_HS_2d_gpu

.PHONY : apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/build

apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/clean:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d && $(CMAKE_COMMAND) -P CMakeFiles/register_HS_2d_gpu.dir/cmake_clean.cmake
.PHONY : apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/clean

apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/depend:
	cd /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/standalone/gpu/registration/2d/CMakeFiles/register_HS_2d_gpu.dir/depend

