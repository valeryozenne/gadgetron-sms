if (WIN32)
    add_definitions(-D__BUILD_GADGETRON_MULTIBAND__)
endif ()

include_directories(
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
    ${CMAKE_SOURCE_DIR}/toolboxes/core
    ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/image
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/algorithm
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/hostutils
    ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
    ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/fft/gpu
    ${CMAKE_SOURCE_DIR}/toolboxes/klt/cpu
    ${CMAKE_SOURCE_DIR}/toolboxes/gadgettools
    ${CMAKE_SOURCE_DIR}/toolboxes/image_io
    ${CMAKE_SOURCE_DIR}/gadgets/mri_core
    ${HDF5_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}/cpp
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${ISMRMRD_INCLUDE_DIR}
)

set( gadgetron_multiband_config_files
    config/Generic_Cartesian_Grappa_EPI_SMS.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Test.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Synthetic.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Synthetic_Post_Process.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Synthetic_Split_Post_Process.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Matlab_Old.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Cplus.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Kill_MB_Scans.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Cplus_Post_Process.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_Cplus_Dicom.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_Read.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_CMRR_Cplus_Info.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_Liryc.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_Liryc_Info.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_WIP_Matlab.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_WIP_Matlab_Post_Process.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_WIP_Matlab_Old.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_WIP_Cplus_Info.xml
    config/Generic_Cartesian_Grappa_EPI_SMS_WIP_Dump.xml
)


add_library(gadgetron_multiband SHARED
gadgetron_multiband_export.h
    ApplySliceOptimalPhaseShiftGadget.cpp
    ApplySliceOptimalPhaseShiftGadget.h
    ApplyWIPSliceOptimalPhaseShiftGadget.cpp
    ApplyWIPSliceOptimalPhaseShiftGadget.h
    BlipACSCalibrationGadget.h
    BlipACSCalibrationGadget.cpp
    BlipACSCalibrationSimpleGadget.h
    BlipACSCalibrationSimpleGadget.cpp
    BlipACSAndSliceCalibrationGadget.h
    BlipACSAndSliceCalibrationGadget.cpp
    PseudoGenericMultibandRecoGadget.h
    PseudoGenericMultibandRecoGadget.cpp
    PseudoWIPGenericMultibandRecoGadget.h
    PseudoWIPGenericMultibandRecoGadget.cpp
    EmptyFlagsGadget.h
    EmptyFlagsGadget.cpp
    CheckMultibandDefoldingGadget.h
    CheckMultibandDefoldingGadget.cpp
    KillMultibandDataGadget.h
    KillMultibandDataGadget.cpp
    CheckWIPMultibandBeforeGadget.h
    CheckWIPMultibandBeforeGadget.cpp
    CheckWIPMultibandDefoldingGadget.h
    CheckWIPMultibandDefoldingGadget.cpp
    RemoveBlipGadget.h
    RemoveBlipGadget.cpp
    WriteLirycSliceCalibrationFlagsGadget.h
    WriteLirycSliceCalibrationFlagsGadget.cpp
    WriteSliceCalibrationFlagsGadget.h
    WriteSliceCalibrationFlagsGadget.cpp
    WriteMultibandInformationGadget.h
    WriteMultibandInformationGadget.cpp
    WriteWIPMultibandInformationGadget.h
    WriteWIPMultibandInformationGadget.cpp
    WriteLirycMultibandInformationGadget.h
    WriteLirycMultibandInformationGadget.cpp
    SMSKspaceGadget.cpp
    SMSKspaceGadget.h
    ${gadgetron_multiband_config_files}
)

set_target_properties(gadgetron_multiband PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})                                                                                                                                                                                                      

target_link_libraries(gadgetron_multiband
    gadgetron_gadgetbase
    gadgetron_toolbox_log
    gadgetron_toolbox_cpucore
    gadgetron_toolbox_gadgettools
    gadgetron_toolbox_image_analyze_io
    ${ISMRMRD_LIBRARIES} 
    ${ACE_LIBRARIES}   
)

 

install(FILES 
    gadgetron_multiband_export.h
    DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(FILES ${gadgetron_multiband_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)

install(TARGETS gadgetron_multiband DESTINATION lib COMPONENT main)
