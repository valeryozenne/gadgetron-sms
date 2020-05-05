if (WIN32)
  add_definitions(-D__BUILD_GADGETRON_GADGET_THERMOCORE__)
  # The two flags below is to fix Windows problems in relation to multiple defined constructors in our headers
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE") 
  set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE") 
endif ()

add_library(gadgetron_thermocore SHARED
    gadgetron_thermocore_export.h
    #ThermoMappingGadget.cpp
    #ThermoMappingGadget.h
    ZeroFillingGadget.cpp
    ZeroFillingGadget.h
    
  )

set_target_properties(gadgetron_thermocore PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_thermocore 
  gadgetron_core
  gadgetron_mricore
  gadgetron_toolbox_log
  gadgetron_toolbox_mri_core
  gadgetron_toolbox_cpucore
  gadgetron_toolbox_cpufft
  gadgetron_toolbox_gpudwt
  gadgetron_toolbox_gpucore
  gadgetron_toolbox_gpusolvers
  gadgetron_toolbox_gpuoperators
  gadgetron_toolbox_gpuparallelmri
)



target_include_directories(gadgetron_thermocore
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})



install(TARGETS gadgetron_thermocore
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

install (FILES  gadgetron_thermocore_export.h                
                DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(config)

set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)
