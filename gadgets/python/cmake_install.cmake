# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/python" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/accumulate_and_recon.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/array_image.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/bucket_recon.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/image_array_recon.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/image_array_recon_rtcine_plotting.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/passthrough.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/passthrough_array_image.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/pseudoreplicagather.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/remove_2x_oversampling.py"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/gadgets/rms_coil_combine.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/config" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/pseudoreplica.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_buckets.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_short.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_image_array.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_image_array_recon.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_passthrough.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/legacy/config/python_short.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/python/config/Generic_Cartesian_Grappa_RealTimeCine_Python.xml"
    )
endif()

