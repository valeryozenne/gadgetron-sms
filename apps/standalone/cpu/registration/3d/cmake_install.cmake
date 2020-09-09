# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/cpu/registration/3d

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu"
         RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/apps/standalone/cpu/registration/3d/register_CK_3d_cpu")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu"
         OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/plplot:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/moco:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/cartesian:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/dicom:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/grappa:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/spiral:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_noncartesian:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/hyper:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/radial:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/pmri:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/thermo_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/sms_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/fatwater:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/epi:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/cmr:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/interventional_mri:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/plplot:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/python:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image_io:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/denoise:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/pattern_recognition:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/ffd:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/nfft:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/nfft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/nfft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/deblur:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fatwater:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_image:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/cmr:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/hyper:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/epi:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/pmri:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri/pmri/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/hostutils:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/math:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/solvers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/solvers/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/solvers/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/gpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/distributed:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/parallel:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/readers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:"
         NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/register_CK_3d_cpu")
    endif()
  endif()
endif()
