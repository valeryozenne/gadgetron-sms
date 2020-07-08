# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so"
         RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/libgadgetron_core.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so"
         OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:"
         NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_core.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gadgetron" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Channel.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Channel.hpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/ChannelIterator.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Message.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Message.hpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/MPMCChannel.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Gadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Context.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Gadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Reader.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Types.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Types.hpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/TypeTraits.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Writer.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/Node.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/LegacyACE.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/PropertyMixin.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/GadgetContainerMessage.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/ChannelAlgorithms.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/variant.hpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/MessageID.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/gadgetron/io" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/io/adapt_struct.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/io/from_string.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/io/ismrmrd_types.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/io/primitives.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/io/primitives.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/config" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/config/distributed_default.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/config/distributed_generic_default.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/config/distributed_image_default.xml"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/readers/cmake_install.cmake")
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers/cmake_install.cmake")
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/parallel/cmake_install.cmake")
  include("/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/distributed/cmake_install.cmake")

endif()

