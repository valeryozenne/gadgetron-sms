
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
  
  
  
  //TODO nb: les trois fonctions ci-dssous ne sont pas très codées, particulièrement la function returnTime
  EXPORTMRICORE char* ReturnTime(int a);  
  EXPORTMRICORE arma::fvec ConversionTimeStamp(uint32_t time_stamp_);
  EXPORTMRICORE arma::fvec DivisionTimeStamp(int a , int b);  

  EXPORTMRICORE int DeleteTemporaryFiles(std::string str_home, std::string str_folder, std::string str_name );
  
  EXPORTMRICORE void Affichage(int image_index, int d, int s, int image_type );
  EXPORTMRICORE void AffichageGT(int image_index, int d, int s, int image_type );
  EXPORTMRICORE void FunctionTest2Void(bool status);
  
  EXPORTMRICORE arma::cx_fvec LoadCplxVectorFromtheDisk(std::string home, std::string path, std::string nom,  std::string str_s,  std::string str_e);
  EXPORTMRICORE arma::fvec LoadVectorFromtheDisk(std::string home, std::string path, std::string nom,  std::string str_s,  std::string str_e);
  EXPORTMRICORE arma::fmat LoadMatrixFromtheDisk(std::string home, std::string path, std::string nom,   std::string str_e);
  EXPORTMRICORE arma::cx_fmat LoadCplxMatrixFromtheDisk(std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e);
  EXPORTMRICORE arma::fmat LoadMatrixFromtheDisk(std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e);
  EXPORTMRICORE arma::fmat LoadMatrixFromtheDisk(std::string home, std::string path, std::string nom, std::string str_d, std::string str_s,  std::string str_e);  
  
  EXPORTMRICORE void SaveIntMatrixOntheDisk(arma::imat MatrixToSave, std::string home, std::string path, std::string nom,  std::string str_e);
  EXPORTMRICORE void SaveMatrixOntheDisk(arma::fmat MatrixToSave, std::string home,std::string path, std::string nom,  std::string str_e);
  EXPORTMRICORE void SaveMatrixOntheDisk(arma::cx_fmat MatrixToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_e);
  EXPORTMRICORE void SaveMatrixOntheDisk(arma::fmat MatrixToSave, std::string home,std::string path, std::string nom, std::string str_d, std::string str_e);
  EXPORTMRICORE void SaveMatrixOntheDisk(arma::fmat MatrixToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e); 
  EXPORTMRICORE void SaveMatrixOntheDisk(arma::cx_fmat MatrixToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e); 
  EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e); 
  EXPORTMRICORE void SaveVectorOntheDisk(arma::cx_fvec VectorToSave, std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e); 
  EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_d,  std::string str_e);
  EXPORTMRICORE void SaveVectorOntheDisk(arma::cx_fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_d,  std::string str_e);
  EXPORTMRICORE void SaveVectorOntheDisk(arma::fvec VectorToSave, std::string home,std::string path, std::string nom, std::string str_e);
  
  EXPORTMRICORE void ShowErrorMessageSaveWithLocation(std::string msg, bool status);
  EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);
  
  EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);
  EXPORTMRICORE void ShowErrorMessageLoadWithLocation(std::string msg, bool status);
  
  
  EXPORTMRICORE std::string IntToString(int value);
  EXPORTMRICORE std::string DoubleToString(double value);
  EXPORTMRICORE std::string DoubleToString1(double value);
  
  EXPORTMRICORE std::string GetHomeDirectory();
 
  EXPORTMRICORE arma::fvec GetInformationFromSpecialCard(ISMRMRD::IsmrmrdHeader h);
  
	

	
  
}
