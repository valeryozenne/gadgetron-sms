if (WIN32)
  add_definitions(-D__BUILD_GADGETRON_GADGET_SMSCORE__)
  # The two flags below is to fix Windows problems in relation to multiple defined constructors in our headers
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE") 
  set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE") 
endif ()

add_library(gadgetron_smscore SHARED
    gadgetron_smscore_export.h
    WriteSingleBandFlagsGadget.cpp
    WriteSingleBandFlagsGadget.h
    EmptyGadget.cpp
    EmptyGadget.h
    GenericReconBase_gadget1of2.h
    GenericReconBase_gadget1of2.cpp
    GenericReconSMSBase.h
    GenericReconSMSBase.cpp
    GenericReconSMSBase_gadget1of2.h
    GenericReconSMSBase_gadget1of2.cpp
    GiveReadoutInformationGadget.h
    GiveReadoutInformationGadget.cpp
    GenericReconSplitSMSGadget.h
    GenericReconSplitSMSGadget.cpp
    GenericReconCheckSizeGadget.cpp
    GenericReconCheckSizeGadget.h
    GenericReconCartesianSliceGrappaGadget.cpp
    GenericReconCartesianSliceGrappaGadget.h
    GenericReconCartesianSliceGrappav3Gadget.cpp
    GenericReconCartesianSliceGrappav3Gadget.h
    GenericReconCartesianSliceGrappavgpuGadget.cpp
    GenericReconCartesianSliceGrappavgpuGadget.h
    GenericReconSMSPrepGadget.cpp
    GenericReconSMSPrepGadget.h
    GenericReconSMSPrepGadget_1of2.cpp
    GenericReconSMSPrepGadget_1of2.h
    GenericReconSMSPostGadget_1of2.cpp
    GenericReconSMSPostGadget_1of2.h
    GenericReconSMSPostGadget.cpp
    GenericReconSMSPostGadget.h
    GenericReconEigenChannelSMSGadget.cpp
    GenericReconEigenChannelSMSGadget.h    
    MessageContainerNumberGadget.cpp
    MultiplyMessageContainerGadget.cpp
    MessageContainerNumberGadget.h
    MultiplyMessageContainerGadget.h
    GenericReconSplitSimpleSMSGadget.h
    GenericReconSplitSimpleSMSGadget.cpp
    GenericReconCartesianReceiveEpiMsgGadget.h
	GenericReconCartesianReceiveEpiMsgGadget.cpp
  )

set_target_properties(gadgetron_smscore PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_smscore
  gadgetron_toolbox_epi
  gadgetron_core
  gadgetron_mricore
  gadgetron_toolbox_log
  gadgetron_toolbox_mri_core
  gadgetron_toolbox_cpucore  
  gadgetron_toolbox_gpudwt
  gadgetron_toolbox_gpucore
  gadgetron_toolbox_gpusolvers
  gadgetron_toolbox_gpuoperators
  gadgetron_toolbox_gpuparallelmri
)

#target_link_libraries(gadgetron_smscore gadgetron_toolbox_gpuparallelmri)

target_include_directories(gadgetron_smscore
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})



install(TARGETS gadgetron_smscore
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

install (FILES  gadgetron_smscore_export.h 
                WriteSingleBandFlagsGadget.cpp
                WriteSingleBandFlagsGadget.h
                GenericReconSMSBase.h
                GenericReconSMSBase.cpp
                EmptyGadget.cpp
                EmptyGadget.h
                GenericReconSplitSMSGadget.h
                GenericReconSplitSimpleSMSGadget.h
				GenericReconCartesianReceiveEpiMsgGadget.h
                DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(config)

set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)
