
/** \file   mri_core_grappa.cpp
    \brief  GRAPPA implementation for 2D and 3D MRI parallel imaging
    \author Hui Xue

    References to the implementation can be found in:

    Griswold MA, Jakob PM, Heidemann RM, Nittka M, Jellus V, Wang J, Kiefer B, Haase A. 
    Generalized autocalibrating partially parallel acquisitions (GRAPPA). 
    Magnetic Resonance in Medicine 2002;47(6):1202-1210.

    Kellman P, Epstein FH, McVeigh ER. 
    Adaptive sensitivity encoding incorporating temporal filtering (TSENSE). 
    Magnetic Resonance in Medicine 2001;45(5):846-852.

    Breuer FA, Kellman P, Griswold MA, Jakob PM. .
    Dynamic autocalibrated parallel imaging using temporal GRAPPA (TGRAPPA). 
    Magnetic Resonance in Medicine 2005;53(4):981-985.

    Saybasili H., Kellman P., Griswold MA., Derbyshire JA. Guttman, MA. 
    HTGRAPPA: Real-time B1-weighted image domain TGRAPPA reconstruction. 
    Magnetic Resonance in Medicine 2009;61(6): 1425-1433. 
*/

#include "mri_core_slice_grappa.h"
#include "mri_core_utility.h"
#include "hoMatrix.h"
#include "hoNDArray_linalg.h"
#include "hoNDFFT.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"

#ifdef USE_OMP
    #include "omp.h"
#endif // USE_OMP

namespace Gadgetron
{

arma::cx_fvec LoadCplxVectorFromtheDisk(std::string home,std::string path,  std::string nom,  std::string str_s,  std::string str_e)
{
    std::string load_data =  home + "/" + path +nom + str_s + str_e;
    arma::cx_fvec vector_to_load;
    bool status = vector_to_load.load(load_data);
    ShowErrorMessageLoadWithLocation(load_data, status);
    return vector_to_load;
}

/// Save desired data on the disk on a specific file
void SaveVectorOntheDisk(arma::fvec VectorToSave,  std::string home, std::string path, std::string nom,  std::string str_e)
{
    std::string save_data = home + "/" + path + nom  +  str_e;
    // 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;
    if (str_e==".bin")
    {
        status = VectorToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
    else if (str_e==".dat")
    {
        status =VectorToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
}

/// Save desired data on the disk on a specific file
void SaveVectorOntheDisk(arma::fvec VectorToSave,  std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e)
{
    std::string save_data = home + "/" + path + nom  + str_d + str_e;
    // 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;
    if (str_e==".bin")
    {
        status = VectorToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
    else if (str_e==".dat")
    {
        status =VectorToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
}



/// Save desired data on the disk on a specific file
void SaveVectorOntheDisk(arma::cx_fvec VectorToSave,  std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e)
{
    std::string save_data = home + "/" + path + nom  + str_d + str_e;
    // 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;
    if (str_e==".bin")
    {
        status = VectorToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
    else if (str_e==".dat")
    {
        status =VectorToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data,status);
    }
}

void SaveVectorOntheDisk(arma::fvec VectorToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e)
{

    std::string save_data = home + "/" + path + nom  + str_d + "_"+ str_s + str_e;
    //std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;

    if (str_e==".bin")
    {
        status = VectorToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }
    else if (str_e==".dat")
    {
        status =VectorToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }

}


void SaveVectorOntheDisk(arma::cx_fvec VectorToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e)
{

    std::string save_data = home + "/" + path + nom  + str_d + "_"+ str_s + str_e;
    //std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;

    if (str_e==".bin")
    {
        status = VectorToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }
    else if (str_e==".dat")
    {
        status =VectorToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }

}

void ShowErrorMessageLoadWithLocation(std::string msg, bool status)
{

    if(status != true)
    {
        std::cout << " " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
        std::cout << "	!!!!  H O U S T O N ,   O N  A  U N  P R O B L E M E  !!!! " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;

        std::cout << "	Problem with loading  :-( " << std::endl;
        std::cout <<  "	Loading data from the disk: "<< msg  <<std::endl;
    }
}


void ShowErrorMessageSaveWithLocation(std::string msg, bool status)
{

    if(status != true)
    {
        std::cout << " " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
        std::cout << "	!!!!  H O U S T O N ,   O N  A  U N  P R O B L E M E  !!!! " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;

        std::cout << "	Problem with saving  :-( " << std::endl;
        std::cout <<  "	Saving data on the disk: "<< msg  <<std::endl;
    }
}



}
