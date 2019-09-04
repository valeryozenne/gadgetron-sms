
/** \file   mri_core_grappa.h
    \brief  GRAPPA implementation for 2D and 3D MRI parallel imaging
    \author Hui Xue
*/

#pragma once

#include "mri_core_export.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"

namespace Gadgetron {

EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e);
EXPORTMRICORE void SaveVectorOntheDisk(arma::cx_fvec VectorToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e);
EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_d,  std::string str_e);
EXPORTMRICORE void SaveVectorOntheDisk(arma::cx_fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_d,  std::string str_e);
EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_e);

EXPORTMRICORE void ShowErrorMessageSaveWithLocation(std::string msg, bool status);
EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);

EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);
EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);

EXPORTMRICORE arma::cx_fvec LoadCplxVectorFromtheDisk(std::string home, std::string path, std::string nom,  std::string str_s,  std::string str_e);


}
