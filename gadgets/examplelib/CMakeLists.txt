if (WIN32)
    add_definitions(-D__BUILD_GADGETRON_EXAMPLELIB__)
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
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${ISMRMRD_INCLUDE_DIR}
    ${HDF5_C_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
)   

if (ARMADILLO_FOUND)
     list(APPEND OPTIMIZED_GADGET_LIBS gadgetron_toolbox_cpucore_math ${ARMADILLO_LIBRARIES})
elseif (ARMADILLO_FOUND)
    message("Armadillo not found, i.e. not compiling Blas/Lapack optimized MRI Gadgets")
endif ()

set( gadgetron_examplelib_header_files ThresholdGadget.h
GenericReconBase.h
GenericEmptyTestGadget.h
EmptyRemoveROOversamplingGadget.h
GenericCheckSizeGadget.h
GenericCheckSizeSimpleGadget.h
GenericDecoupeGadget.h
GenericConcateneGadget.h
GenericChangeSizeTo8Gadget.h
GenericChangeSizeTo7Gadget.h
GenericReconCartesianMultibandPrepGadget.h
GenericReconCartesianMultibandSliceGrappaGadget.h
GenericReconCartesianMultibandPostGadget.h
GenericReconGadget.h
GenericCheckSizeDataRefSBGadget.h
GenericMBReconDataBase.h
hoNDArrayToCuNDArrayGadget.h
CuNDArrayTohoNDArrayGadget.h
#SegmentedAquisitionGadget.h
)

set( gadgetron_examplelib_src_files ThresholdGadget.cpp
  GenericEmptyTestGadget.cpp
GenericReconBase.cpp
EmptyRemoveROOversamplingGadget.cpp
GenericCheckSizeGadget.cpp
GenericCheckSizeSimpleGadget.cpp
GenericDecoupeGadget.cpp
GenericConcateneGadget.cpp
GenericChangeSizeTo8Gadget.cpp
GenericChangeSizeTo7Gadget.cpp
GenericReconCartesianMultibandPrepGadget.cpp
GenericReconCartesianMultibandSliceGrappaGadget.cpp
GenericReconCartesianMultibandPostGadget.cpp
GenericCheckSizeDataRefSBGadget.cpp
GenericMBReconDataBase.cpp
hoNDArrayToCuNDArrayGadget.cpp
CuNDArrayTohoNDArrayGadget.cpp
#SegmentedAquisitionGadget.cpp
)

set( gadgetron_examplelib_config_files
    config/threshold.xml
    config/Generic_threshold.xml
    config/Generic_Cartesian_Grappa_EPI_Cut_And_Combine.xml
    config/Generic_Change_Size.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Franck.xml
    config/Generic_Cartesian_Multiband.xml
    config/Generic_Cartesian_Multiband_Prep.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Slice_Grappa_Franck.xml
    config/Test_change_ho_to_cuNDArray.xml
    config/Generic_Cartesian_Multiband_Prep_Post.xml
)

add_library(gadgetron_examplelib SHARED 
    gadgetron_examplelib_export.h 
    ${gadgetron_examplelib_header_files} 
    ${gadgetron_examplelib_src_files}
    ${gadgetron_examplelib_config_files}
    ${OPTIMIZED_GADGETS}
    ${OPTIMIZED_GADGET_HEADERS}
)

set_target_properties(gadgetron_examplelib PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_examplelib 
    gadgetron_gadgetbase
    gadgetron_toolbox_log
    gadgetron_toolbox_cpucore
    gadgetron_toolbox_cpufft
    gadgetron_toolbox_image_analyze_io
    gadgetron_toolbox_hostutils
    ${ISMRMRD_LIBRARIES} 
    ${FFTW3_LIBRARIES} 
    optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY} 
    ${OPTIMIZED_GADGET_LIBS}
    ${Boost_LIBRARIES}
)
if(ARMADILLO_FOUND)
   target_link_libraries(gadgetron_examplelib
    gadgetron_toolbox_cpuoperator
    gadgetron_toolbox_cpuklt
    gadgetron_toolbox_mri_core
)
endif()

if (ZFP_FOUND)
   target_link_libraries(gadgetron_examplelib ${ZFP_LIBRARIES})
endif ()

install(FILES 
    gadgetron_examplelib_export.h
    ${gadgetron_examplelib_header_files}
    ${OPTIMIZED_GADGET_HEADERS}
    DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

if (ARMADILLO_FOUND)
    install(FILES ${gadgetron_examplelib_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
endif ()

install(TARGETS gadgetron_examplelib DESTINATION lib COMPONENT main)
