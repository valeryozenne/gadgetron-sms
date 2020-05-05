if (ARMADILLO_FOUND)
   install (FILES     
    Generic_Cartesian_Grappa_EPI_Thermo_4.0.xml
    DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main
    )
elseif (ARMADILLO_FOUND)
  #message("Armadillo not found, only unoptimized generic trajectory config files will be available")
endif ()

#install (FILES 
#  generic_gpusense_cg_unoptimized.xml 
#  generic_gpusense_sb_unoptimized.xml 
#  DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH})
