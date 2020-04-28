if (WIN32)
    add_definitions(-D__BUILD_GADGETRON_SMSCORE__)
endif ()

find_package(ZFP)
if (ZFP_FOUND)
    message("ZFP Found")
    add_definitions(-DGADGETRON_COMPRESSION_ZFP)
    include_directories(${ZFP_INCLUDE_DIR})
else ()
    message("ZFP NOT Found")
endif ()


if (MKL_FOUND)
    # This is a fix for the bug in SVD when MKL is multi-threaded
    message("MKL Found, enabling MKL for mri_core gadgets.")
    add_definitions(-DHAVE_MKL)
    # These are needed to get the linking to work properly when
    # MKL is installed, but Armadillo is NOT using it.
    list(APPEND EXTRA_MKL_LIBRARIES mkl_core)
    list(APPEND EXTRA_MKL_LIBRARIES mkl_intel_thread)
    include_directories( ${MKL_INCLUDE_DIR} )
    link_directories( ${MKL_LIB_DIR} ${MKL_COMPILER_LIB_DIR} )
endif ()

find_package(HDF5 1.8 REQUIRED COMPONENTS C)

include_directories(
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
    ${CMAKE_SOURCE_DIR}/toolboxes/core
    ${CMAKE_SOURCE_DIR}/toolboxes/log
    ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/image
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/algorithm
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/hostutils
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
    ${CMAKE_SOURCE_DIR}/toolboxes/dwt/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/operators
    ${CMAKE_SOURCE_DIR}/toolboxes/operators/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/solvers 
    ${CMAKE_SOURCE_DIR}/toolboxes/solvers/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
    ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/klt/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/gadgettools
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/transformation
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/solver
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/warper
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/dissimilarity
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/register
    ${CMAKE_SOURCE_DIR}/toolboxes/registration/optical_flow/cpu/application
    ${CMAKE_SOURCE_DIR}/toolboxes/image_io
    ${CMAKE_SOURCE_DIR}/toolboxes/mri_image
    ${CMAKE_SOURCE_DIR}/toolboxes/denoise
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${ISMRMRD_INCLUDE_DIR}
    ${HDF5_C_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
)   

if (ARMADILLO_FOUND)
 #   list(APPEND OPTIMIZED_GADGETS NoiseAdjustGadget.cpp)
 #   list(APPEND OPTIMIZED_GADGETS PCACoilGadget.cpp)
 #   list(APPEND OPTIMIZED_GADGET_HEADERS NoiseAdjustGadget.h)
 #   list(APPEND OPTIMIZED_GADGET_HEADERS PCACoilGadget.h)
    list(APPEND OPTIMIZED_GADGET_LIBS gadgetron_toolbox_cpucore_math ${ARMADILLO_LIBRARIES})
elseif (ARMADILLO_FOUND)
    message("Armadillo not found, i.e. not compiling Blas/Lapack optimized MRI Gadgets")
endif ()

set( gadgetron_smscore_header_files  	
        WriteSliceCalibrationFlagsGadget.h
        GenericReconCheckSizeGadget.h        
        GenericReconBase.h
        GenericReconGadget.h
        GadgetIsmrmrdReadWrite.h
        NHLBICompression.h
        ImageArraySendMixin.h        
        ImageArraySendMixin.hpp
        GenericReconSMSBase.h
        GenericReconSMSPrepGadget.h
        GenericReconSMSPostGadget.h
        GenericReconCartesianSliceGrappaGadget.h
        GenericReconCartesianSliceGrappav3Gadget.h
        GenericReconCartesianSliceGrappavgpuGadget.h
        GenericReconEigenChannelSMSGadget.h
)

set( gadgetron_smscore_src_files 	
        WriteSliceCalibrationFlagsGadget.cpp
        GenericReconCheckSizeGadget.cpp        
        GenericReconBase.cpp
        GenericReconGadget.cpp
        GadgetIsmrmrdReadWrite.cpp
        GenericReconSMSBase.cpp
        GenericReconSMSPrepGadget.cpp
        GenericReconSMSPostGadget.cpp
        GenericReconCartesianSliceGrappaGadget.cpp  
        GenericReconCartesianSliceGrappav3Gadget.cpp
        GenericReconCartesianSliceGrappavgpuGadget.cpp
        GenericReconEigenChannelSMSGadget.cpp
        )

set( gadgetron_smscore_config_files 
    config/Generic_Cartesian_Slice_Grappa_EPI_v2.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_v3.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_v4.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_v5.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_vopenmp.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_vgpu.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_vgpu_gold.xml
    config/Generic_Cartesian_Slice_Grappa_EPI_vcpu.xml
)

add_library(gadgetron_smscore SHARED 
    gadgetron_smscore_export.h 
    ${gadgetron_smscore_header_files} 
    ${gadgetron_smscore_src_files}
    ${gadgetron_smscore_config_files}
    ${OPTIMIZED_GADGETS}
    ${OPTIMIZED_GADGET_HEADERS}
        )
# Above: Whitespace misaligned, and one file on each line please.

set_target_properties(gadgetron_smscore PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_smscore 
    gadgetron_gadgetbase
    gadgetron_toolbox_log
    gadgetron_toolbox_cpucore
    gadgetron_toolbox_cpufft
    gadgetron_toolbox_gpufft    
    gadgetron_toolbox_gpuparallelmri
    gadgetron_toolbox_image_analyze_io
    gadgetron_toolbox_denoise
    gadgetron_toolbox_hostutils
    ${ISMRMRD_LIBRARIES} 
    ${FFTW3_LIBRARIES} 
    optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY} 
    ${OPTIMIZED_GADGET_LIBS}
    ${Boost_LIBRARIES}
)
if(ARMADILLO_FOUND)
   target_link_libraries(gadgetron_smscore
    gadgetron_toolbox_cpuoperator
    gadgetron_toolbox_cpuklt
    gadgetron_toolbox_mri_core
)
endif()

if (ZFP_FOUND)
   target_link_libraries(gadgetron_smscore ${ZFP_LIBRARIES})
endif ()

install(FILES 
    gadgetron_smscore_export.h
    ${gadgetron_smscore_header_files}
    ${OPTIMIZED_GADGET_HEADERS}
    DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

if (ARMADILLO_FOUND)
    install(FILES ${gadgetron_smscore_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
endif ()

install(TARGETS gadgetron_smscore DESTINATION lib COMPONENT main)
