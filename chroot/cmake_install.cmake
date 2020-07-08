# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xscriptsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/chroot" TYPE PROGRAM FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/copy-cuda-lib.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/start-gadgetron.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/enter-chroot-env.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/gadgetron-dependency-query.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/siemens_to_ismrmrd.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/gadgetron_ismrmrd_client.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/gadgetron_ismrmrd_client_noise_summary.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/gt_alive.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/run-gadgetron-dependency-query.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/run-gadgetron_ismrmrd_client.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/run-gadgetron_ismrmrd_client_noise_summary.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/run-gt_alive.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/run-siemens_to_ismrmrd.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/start-env.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/start.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/mount_image.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/start-gadgetron-from-image.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/mount.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/stop.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/umount_image.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/install_chroot_image.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/clean_gadgetron_data.sh"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/chroot/nvidia-copy.sh"
    )
endif()

