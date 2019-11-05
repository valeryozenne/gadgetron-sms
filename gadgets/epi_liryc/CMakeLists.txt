if (WIN32)
  add_definitions(-D__BUILD_GADGETRON_EPI_LIRYC__)
endif ()

include_directories(
  ${CMAKE_SOURCE_DIR}/gadgets/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu/math
  ${CMAKE_SOURCE_DIR}/toolboxes/gadgettools
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/epi
  ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
  ${ARMADILLO_INCLUDE_DIRS}
)

set( gadgetron_epi_liryc_config_files 

)


set( gadgetron_epi_liryc_src_files EPIReconXStudyGadget.cpp  
    EPICorrStudyGadget.cpp
    EPICorrMultiBandCMRRGadget.cpp
    EPICorrMultiBandSiemensSimpleGadget.cpp    
    EPICorrMultiBandSiemensGadget.cpp
    EPICorrMultiBandSimpleGadget.cpp
    IFFTXGadget.cpp
    ReverseLineGadget.cpp
)

set( gadgetron_epi_liryc_header_files EPIReconXStudyGadget.h
    EPICorrStudyGadget.h
    EPICorrMultiBandCMRRGadget.h
    EPICorrMultiBandSiemensSimpleGadget.h
    EPICorrMultiBandSiemensGadget.h
    EPICorrMultiBandSimpleGadget.h
    IFFTXGadget.h
    ReverseLineGadget.h
)

add_library(gadgetron_epi_liryc SHARED 
    gadgetron_epi_liryc_export.h
    ${gadgetron_epi_liryc_header_files} 
    ${gadgetron_epi_liryc_src_files}
    ${gadgetron_epi_liryc_config_files}
)

set_target_properties(gadgetron_epi_liryc PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(
  gadgetron_epi_liryc
  gadgetron_gadgetbase
  gadgetron_toolbox_cpucore
  gadgetron_toolbox_cpufft
  gadgetron_toolbox_cpucore_math
  gadgetron_toolbox_log
  ${ISMRMRD_LIBRARIES}
  optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY} 
)

install(FILES 
    gadgetron_epi_liryc_export.h
     ${gadgetron_epi_liryc_header_files}
  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(TARGETS gadgetron_epi_liryc DESTINATION lib COMPONENT main)

install(FILES
    DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
