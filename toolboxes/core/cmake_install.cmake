# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/benoit/gadgetron_install_dir/gadgetron4_sms/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gadgetron" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/core_defines.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/NDArray.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/NDArray_utils.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/complext.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/vector_td.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/vector_td_operators.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/vector_td_utilities.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/vector_td_io.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/real_utilities.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/GadgetronException.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/GadgetronTimer.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/Gadgetron_enable_types.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/cmake_install.cmake")
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/gpu/cmake_install.cmake")

endif()

