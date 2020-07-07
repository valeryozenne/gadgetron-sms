# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/cmake" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindArmadillo.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindCUDA_advanced.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindDCMTK.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindFFTW3.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindGLEW.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindGperftools.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindMKL.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindNumPy.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindOctave.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindPLplot.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindZFP.cmake"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/cmake/FindCBLAS.cmake"
    )
endif()

