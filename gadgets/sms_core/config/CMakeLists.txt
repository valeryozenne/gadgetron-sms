if (ARMADILLO_FOUND)
  install (FILES 
    #generic_gpusense_cg.xml 
    #generic_gpusense_cg_singleshot.xml 
    #generic_gpusense_sb_singleshot.xml 
    #generic_gpusense_nlcg_singleshot.xml
    #generic_gpu_ktsense_singleshot.xml 
    Generic_Cartesian_Slice_Grappa_EPI.xml
    Generic_Cartesian_Slice_Grappa_EPI_Error.xml
    Generic_Cartesian_Slice_Grappa_EPI_v5.xml
    Generic_Cartesian_Slice_Grappa_EPI_v5_1of2.xml
    Generic_Cartesian_Slice_Grappa_EPI_With_Msg.xml
    Generic_Cartesian_Slice_Grappa_EPI_v2.xml
    Generic_Cartesian_Slice_Grappa_EPI_v3.xml
    Generic_Cartesian_Slice_Grappa_EPI_v4.xml
    Generic_Cartesian_Slice_Grappa_EPI_vcpu.xml
    Generic_Cartesian_Slice_Grappa_EPI_vgpu.xml
    Generic_Cartesian_Slice_Grappa_EPI_vgpu_gold.xml
    Generic_Cartesian_Slice_Grappa_EPI_vgpu_gold_error.xml
    Generic_Cartesian_Slice_Grappa_EPI_vopenmp.xml
    DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main
    )
elseif (ARMADILLO_FOUND)
  #message("Armadillo not found, only unoptimized generic trajectory config files will be available")
endif ()

#install (FILES 
#  generic_gpusense_cg_unoptimized.xml 
#  generic_gpusense_sb_unoptimized.xml 
#  DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH})
