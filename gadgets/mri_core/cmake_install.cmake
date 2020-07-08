# Install script for directory: /home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core

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
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/gadgetron_mricore_export.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GadgetMRIHeaders.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/NoiseAdjustGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/PCACoilGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/RateLimitGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/AcquisitionPassthroughGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/AccumulatorGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/FFTGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/CombineGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/CropAndCombineGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageWriterGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/writers/MRIImageWriter.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/readers/MRIImageReader.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/NoiseAdjustGadget_unoptimized.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ExtractGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/FloatToFixPointGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/RemoveROOversamplingGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/CoilReductionGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ScaleGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/FlowPhaseSubtractionGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/readers/GadgetIsmrmrdReader.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/PhysioInterpolationGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/IsmrmrdDumpGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/AsymmetricEchoAdjustROGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/MaxwellCorrectionGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/CplxDumpGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/dependencyquery/DependencyQueryGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/dependencyquery/DependencyQueryWriter.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ComplexToFloatGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/AcquisitionAccumulateTriggerGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/BucketToBufferGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageArraySplitGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/PseudoReplicatorGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/SimpleReconGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageSortGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconBase.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianFFTGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianGrappaGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianSpiritGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianNonLinearSpirit2DTGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianReferencePrepGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconPartialFourierHandlingGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconPartialFourierHandlingFilterGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconPartialFourierHandlingPOCSGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconKSpaceFilteringGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconFieldOfViewAdjustmentGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconImageArrayScalingGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconEigenChannelGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconNoiseStdMapComputingGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericImageReconGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconAccumulateImageTriggerGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericImageReconArrayToImageGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconReferenceKSpaceDelayedBufferGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/WhiteNoiseInjectorGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageFinishGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/dependencyquery/NoiseSummaryGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/NHLBICompression.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageAccumulatorGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/writers/GadgetIsmrmrdWriter.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageResizingGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageArraySendMixin.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/ImageArraySendMixin.hpp"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/DenoiseGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/CoilComputationGadget.h"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/GenericReconCartesianGrappaAIGadget.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gadgetron/config" TYPE FILE FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/default.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/default_short.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/default_optimized.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/default_measurement_dependencies.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/default_measurement_dependencies_ismrmrd_storage.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/isalive.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/gtquery.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_FFT.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_FFT_python.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_FFT_EPI.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_FFT_EPI_python.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_SNR.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_T2W.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_RealTimeCine.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_RealTimeCine_Cloud.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_EPI.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_EPI_AVE.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_EPI_Msg.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Spirit.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Spirit_RealTimeCine.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Spirit_SASHA.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_NonLinear_Spirit_RealTimeCine.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_RandomSampling_NonLinear_Spirit_RealTimeCine.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_NonLinear_Spirit_RealTimeCine_Cloud.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_RandomSampling_NonLinear_Spirit_RealTimeCine_Cloud.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_Cine_Denoise.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/NoiseSummary.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/ismrmrd_dump.xml"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/config/Generic_Cartesian_Grappa_AI.xml"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so.4.1.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so.4.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/libgadgetron_mricore.so.4.1.1"
    "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/libgadgetron_mricore.so.4.1"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so.4.1.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so.4.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/denoise:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/hostutils:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/python:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image_io:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/math:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:"
           NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so"
         RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/gadgets/mri_core/libgadgetron_mricore.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so"
         OLD_RPATH "/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core/writers:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/denoise:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/hostutils:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/operators/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/python:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/registration/optical_flow/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/dwt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/mri_core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/core:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/fft/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/image_io:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/klt/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu/math:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/core/cpu:/home/benoit/gadgetron_install_dir/gadgetron4_sms/mrprogs/gadgetron/toolboxes/log:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial:"
         NEW_RPATH ".:/home/benoit/gadgetron_install_dir/gadgetron4_sms/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgadgetron_mricore.so")
    endif()
  endif()
endif()

