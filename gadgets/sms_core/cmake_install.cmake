# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so.4.1.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so.4.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/libgadgetron_smscore.so.4.1.1"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/libgadgetron_smscore.so.4.1"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so.4.1.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so.4.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/pmri/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/denoise:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/hostutils:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/python:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image_io:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/solvers/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/math:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/nfft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/cuda-10.0/lib64:"
           NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so"
         RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/libgadgetron_smscore.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so"
         OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/pmri/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/denoise:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/hostutils:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/python:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image_io:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/solvers/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/math:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/nfft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/local/cuda-10.0/lib64:"
         NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_smscore.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gadgetron" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/gadgetron_smscore_export.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/WriteSingleBandFlagsGadget.cpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/WriteSingleBandFlagsGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/GenericReconSMSBase.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/GenericReconSMSBase.cpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/EmptyGadget.cpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/EmptyGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/EmptyGadget_GPUBenoit.cpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/EmptyGadget_GPUBenoit.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/GenericReconSplitSMSGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/GenericReconSplitSimpleSMSGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/GenericReconCartesianReceiveEpiMsgGadget.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core/config/cmake_install.cmake")

endif()

