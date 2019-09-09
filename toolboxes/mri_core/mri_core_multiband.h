
/** \file   mri_core_utility_interventional.cpp
    \brief  Implementation useful utility functionalities for thermometry
    \author Valery Ozenne 
*/

#pragma once

#include "mri_core_export.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"

using namespace std;

namespace Gadgetron
{   
  

 EXPORTMRICORE arma::fvec get_information_from_multiband_special_card(ISMRMRD::IsmrmrdHeader h);

 EXPORTMRICORE arma::fvec get_information_from_wip_multiband_special_card(ISMRMRD::IsmrmrdHeader h);

 EXPORTMRICORE void save_kspace_data(arma::cx_fcube input,std::string str_home ,string str_folder ,  std::string name);

 EXPORTMRICORE arma::ivec map_interleaved_acquisitions(int number_of_slices, bool no_reordering );

 EXPORTMRICORE arma::imat get_map_slice_single_band(int MB_factor, int lNumberOfStacks, arma::ivec order_of_acquisition_mb, bool no_reordering);

 EXPORTMRICORE arma::ivec map_interleaved_acquisitions_mb(int number_of_slices, bool no_reordering );



}
