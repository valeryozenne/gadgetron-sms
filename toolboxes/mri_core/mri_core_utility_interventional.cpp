
/** \file   mri_core_utility_interventional.cpp
    \brief  Implementation useful utility functionalities for thermometry
    \author Valery Ozenne 
*/

#include "mri_core_utility_interventional.h"
#include "hoNDArray_elemwise.h"
#include "hoArmadillo.h"
#include <pwd.h>

#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"

using namespace std;

namespace Gadgetron
{

  
  
int DeleteTemporaryFiles(std::string str_home, std::string str_folder, std::string str_name )
{
    std::string str_exec = "exec rm ";

    std::string str_delete =str_exec + str_home + "/" + str_folder + str_name ;

    const char *cmd_delete = str_delete.c_str();

    int value_returned = std::system(cmd_delete);

    if (system(NULL)) {

        GDEBUG("All temporary files deleted  \n" );

    }
    else
    {
          GDEBUG("Files have not been deleted  \n" );
    }

    return 0;

}
  
  
  
  
arma::fvec GetInformationFromSpecialCard(ISMRMRD::IsmrmrdHeader h)
{
	arma::fvec output(13);
	output.zeros();

	if (h.userParameters)
	{
		for (std::vector<ISMRMRD::UserParameterDouble>::const_iterator i (h.userParameters->userParameterDouble.begin());
				i != h.userParameters->userParameterDouble.end(); i++)
		{
			if (i->name == "isZeroFilling") {
				output(0) = i->value;
				GDEBUG(" Special Card Instructions 'doZeroFilling_' found with value equal to %f \n", output(0) );
			}
			else if (i->name == "isMOCO") {
				output(1) = i->value;
				GDEBUG(" Special Card Instructions 'doMOCO_' found with value equal to %f \n", output(1) );
			}
			else if (i->name == "isThermo") {
				output(2) = i->value;
				GDEBUG(" Special Card Instructions 'doThermometry_' found with value equal to %f \n", output(2) );
			}
			else if (i->name == "thermoMethod") {
				output(3) = i->value;
				GDEBUG(" Special Card Instructions 'thermoMethod' found with value equal to %f \n", output(3) );
			}
			else if (i->name == "atlasNb") {
				output(4) = i->value;
				GDEBUG(" Special Card Instructions 'atlasNb' found with value equal to %f \n", output(4) );
			}
			else if (i->name == "repNb") {
				output(5) = i->value;
				GDEBUG(" Special Card Instructions 'repNb' found with value equal to %f \n", output(5)  );
			}			
			else if (i->name == "isBaseLineCorrection") {
				output(6) = i->value;
				GDEBUG(" Special Card Instructions 'isBaseLineCorrection' found with value equal to %f \n", output(6)  );
			}
			else if (i->name == "isTemporalFiltering") {
				output(7) = i->value;
				GDEBUG(" Special Card Instructions 'isTemporalFiltering_' found with value equal to %f \n", output(7)  );
			}
			else if (i->name == "isThermoguide") {
				output(8) = i->value;
				GDEBUG(" Special Card Instructions 'isThermoguide' found with value equal to %f \n", output(8)  );
			}
			else if (i->name == "isSendThermo") {
				output(9) = i->value;
				GDEBUG(" Special Card Instructions 'isSendThermo' found with value equal to %f \n", output(9)  );
			}			
			else if (i->name == "isPID") 
			{
				output(10) = i->value;
				GDEBUG(" Special Card Instructions 'doPID' found with value equal to %f \n", output(10)  );
			}
			else if (i->name == "delay") 
			{
				output(10) = i->value;
				GDEBUG(" Special Card Instructions 'delay' found with value equal to %f \n", output(10)  );
			}
			else if (i->name == "duration") 
			{
				output(11) = i->value;
				GDEBUG(" Special Card Instructions 'duration' found with value equal to %f \n", output(11)  );
			}
			else if (i->name == "power") 
			{
				output(12) = i->value;
				GDEBUG(" Special Card Instructions 'power' found with value equal to %f \n", output(12)  );
			}
			else {
				GDEBUG("WARNING: unused user parameter parameter %s found\n", i->name.c_str());
			}
		}
	} else {
		GDEBUG("Special Card Instructions are supposed to be in the UserParameters. No user parameter section found\n");
		//return GADGET_OK;
	}
	
		
	return output;
}
  
  
  
  
  
  
  
char* ReturnTime(int a)
{

	int acquisition_hour =	a;
	
	std::ostringstream io_acquisition_;
	io_acquisition_ << acquisition_hour;
	
	std::string str_hour;  
	

	if (a < 10)
	{
	
	str_hour= std::string("0")+io_acquisition_.str();
	
	}
	else
	{
	  str_hour= io_acquisition_.str();
	}
	 
	
	char* char_type_hour = new char[str_hour.length()];
	strcpy(char_type_hour, str_hour.c_str());

	
	
	

	return char_type_hour;
}
  
  
arma::fvec DivisionTimeStamp(int a , int b)
{
	int reste = a % b;
	int produit = ( a - reste ) / b ;

	arma::fvec output(2);

	output(0)=produit;
	output(1)=reste;

	return output;
}
  
  
arma::fvec ConversionTimeStamp(uint32_t time_stamp_)
{

	arma::fvec time_stamp_int_ = DivisionTimeStamp(time_stamp_,4);

	arma::fvec hours    = DivisionTimeStamp(time_stamp_int_(0),60*60*100);
	arma::fvec minutes  = DivisionTimeStamp(hours(1),60*100);
	arma::fvec seconds  = DivisionTimeStamp(minutes(1),100);

	arma::fvec output(5);
	output.zeros();

	output(0) = hours(0);
	output(1) = minutes(0);
	output(2) = seconds(0);
	output(3) = seconds(1);
	output(4) = time_stamp_int_(1);

	return output;
}

void AffichageGT(int image_index, int d, int s, int image_type )
{
 if (image_type==1) 
 GDEBUG(" image_index : %d     | dynamic : %d     | slice : %d    | image_type  :  MAGNITUDE   \n", image_index, d, s );
 else if (image_type==2)
 GDEBUG(" image_index : %d     | dynamic : %d     | slice : %d    | image_type  :  PHASE       \n", image_index, d, s ); 
}
  
void Affichage(int image_index, int d, int s, int image_type )
{

	std::cout <<  " ----------------------------------------------------------------------------------" << std::endl;
	std::cout <<  "| image_index : "<<  image_index ;
	std::cout <<  "    | dynamic : "<<  d ;
	std::cout <<  "    | slice : "<<  s ;
	if (image_type==1)
		std::cout <<  "   | image_type  :  MAGNITUDE   |"<< std::endl;
	else if (image_type==2)
		std::cout <<  "   | image_type  :  PHASE       |"<< std::endl;
	std::cout <<  " ----------------------------------------------------------------------------------" << std::endl;

}
  
void FunctionTest2Void(bool status)
{

	if(status != true)
	{
		std::cout << " " << std::endl;		
		std::cout << "	Non non  :-( " << std::endl;		
	}
	else
	{
		std::cout << " " << std::endl;		
		std::cout << "	Oui oui :-) " << std::endl;
	}  
	
}


std::string IntToString(int value)
{

	std::ostringstream os;
	os << value;
	std::string s_value = os.str();

	return s_value;

}

std::string DoubleToString(double value)
{

	std::ostringstream os;
	os << value;
	std::string s_value = os.str();	

	return s_value;

}

std::string DoubleToString1(double value)
{

	std::ostringstream os;
	os << roundf(value*10)/10;
	std::string s_value = os.str();

	return s_value;

}


void SaveIntMatrixOntheDisk(arma::imat MatrixToSave, std::string home,  std::string path, std::string nom,  std::string str_e)
{

	std::string save_data =  home + "/" +  path + nom  + str_e;
	// 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
	bool status;

	if (str_e==".bin")
	{
		status = MatrixToSave.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data,status);
	}
	else if (str_e==".dat")
	{
		status =MatrixToSave.save(save_data, arma::raw_ascii);
		ShowErrorMessageSaveWithLocation(save_data,status);
	}

}


arma::fmat LoadMatrixFromtheDisk(std::string home,  std::string path, std::string nom,   std::string str_e)
{

	std::string load_data = home + "/" + path +nom +  str_e;
	// 	    std::cout <<  "	Load data from the disk: "<< load_data  <<std::endl;

	arma::fmat matrix_to_load;
	bool status = matrix_to_load.load(load_data);
	ShowErrorMessageLoadWithLocation(load_data, status);

	return matrix_to_load;
}

arma::fmat LoadMatrixFromtheDisk(std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e)
{

	std::string load_data =home + "/" + path +nom + str_d + str_e;
	// 	    std::cout <<  "	Load data from the disk: "<< load_data  <<std::endl;

	arma::fmat matrix_to_load;
	bool status = matrix_to_load.load(load_data);
	ShowErrorMessageLoadWithLocation(load_data, status);

	return matrix_to_load;
}


arma::cx_fmat LoadCplxMatrixFromtheDisk(std::string home, std::string path, std::string nom, std::string str_d,  std::string str_e)
{

    std::string load_data =home + "/" + path +nom + str_d + str_e;
    // 	    std::cout <<  "	Load data from the disk: "<< load_data  <<std::endl;

    arma::cx_fmat matrix_to_load;
    bool status = matrix_to_load.load(load_data);
    ShowErrorMessageLoadWithLocation(load_data, status);

    return matrix_to_load;
}

/// Get data from a file on the disk
arma::fvec LoadVectorFromtheDisk(std::string home,std::string path,  std::string nom,  std::string str_s,  std::string str_e)
{
	std::string load_data =  home + "/" + path +nom + str_s + str_e;
	arma::fvec vector_to_load;
	bool status = vector_to_load.load(load_data);
	ShowErrorMessageLoadWithLocation(load_data, status);
	return vector_to_load;
}


arma::cx_fvec LoadCplxVectorFromtheDisk(std::string home,std::string path,  std::string nom,  std::string str_s,  std::string str_e)
{
    std::string load_data =  home + "/" + path +nom + str_s + str_e;
    arma::cx_fvec vector_to_load;
    bool status = vector_to_load.load(load_data);
    ShowErrorMessageLoadWithLocation(load_data, status);
    return vector_to_load;
}

arma::fmat LoadMatrixFromtheDisk(  std::string home,std::string path, std::string nom, std::string str_d, std::string str_s,  std::string str_e)
{

	std::string load_data =  home + "/" + path +nom + str_d + "_"+ str_s + str_e;
	// 	    std::cout <<  "	Load data from the disk: "<< load_data  <<std::endl;

	arma::fmat matrix_to_load;
	bool status = matrix_to_load.load(load_data);
	ShowErrorMessageLoadWithLocation(load_data, status);

	return matrix_to_load;
}

void SaveMatrixOntheDisk(arma::fmat MatrixToSave, std::string home,  std::string path, std::string nom,  std::string str_e)
{

	std::string save_data = home + "/" + path + nom  +  str_e;
	// 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
	bool status;

	if (str_e==".bin")
	{
		status = MatrixToSave.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".dat")
	{
		status =MatrixToSave.save(save_data, arma::raw_ascii);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".pgm")
	{	  
		status =MatrixToSave.save(save_data, arma::pgm_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	/*else if (str_e==".png")
	{
		arma::Mat<unsigned char> MatriceChar(size(MatrixToSave,0),size(MatrixToSave,1));
		MatriceChar.ones();
	  
		status =MatriceChar.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
		
		
		
// 	}*/
	

}


void SaveMatrixOntheDisk(arma::fmat MatrixToSave, std::string home,  std::string path, std::string nom, std::string str_d, std::string str_e)
{

	std::string save_data = home + "/" + path + nom  + str_d + str_e;
	// 	  std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
	bool status;

	if (str_e==".bin")
	{
		status = MatrixToSave.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".dat")
	{
		status =MatrixToSave.save(save_data, arma::raw_ascii);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".pgm")
	{	  
		status =MatrixToSave.save(save_data, arma::pgm_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".png")
	{
		arma::Mat<unsigned char> MatriceChar(size(MatrixToSave,0),size(MatrixToSave,1));
		MatriceChar.ones();
	  
		status =MatriceChar.save(save_data, arma::raw_binary);

		ShowErrorMessageSaveWithLocation(save_data, status);

	}

}



void SaveMatrixOntheDisk(arma::fmat MatrixToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e)
{

	std::string save_data = home + "/" + path + nom  + str_d + "_"+ str_s + str_e;
	//std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
	bool status;

	
	
	if (str_e==".bin")
	{
		status = MatrixToSave.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".dat")
	{
		status =MatrixToSave.save(save_data, arma::raw_ascii);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".pgm")
	{	  
	  
		status =MatrixToSave.save(save_data, arma::pgm_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".png")
	{
		arma::Mat<unsigned char> MatriceChar(size(MatrixToSave,0),size(MatrixToSave,1));
		MatriceChar.ones();
	  
		status =MatriceChar.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
		
		
		
	}

}




void SaveMatrixOntheDisk(arma::cx_fmat MatrixToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_e)
{

    std::string save_data = home + "/" + path + nom  + str_d  + str_e;
    //std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
    bool status;



    if (str_e==".bin")
    {
        status = MatrixToSave.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }
    else if (str_e==".dat")
    {
        status =MatrixToSave.save(save_data, arma::raw_ascii);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }
    else if (str_e==".pgm")
    {

        status =MatrixToSave.save(save_data, arma::pgm_binary);
        ShowErrorMessageSaveWithLocation(save_data, status);
    }
    else if (str_e==".png")
    {
        arma::Mat<unsigned char> MatriceChar(size(MatrixToSave,0),size(MatrixToSave,1));
        MatriceChar.ones();

        status =MatriceChar.save(save_data, arma::raw_binary);
        ShowErrorMessageSaveWithLocation(save_data, status);



    }

}


void SaveMatrixOntheDisk(arma::cx_fmat MatrixToSave,  std::string home, std::string path, std::string nom, std::string str_d, std::string str_s, std::string str_e)
{

	std::string save_data = home + "/" + path + nom  + str_d + "_"+ str_s + str_e;
	//std::cout <<  "	Save data on the disk: "<< save_data  <<std::endl;
	bool status;

	
	
	if (str_e==".bin")
	{
		status = MatrixToSave.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".dat")
	{
		status =MatrixToSave.save(save_data, arma::raw_ascii);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".pgm")
	{	  
	  
		status =MatrixToSave.save(save_data, arma::pgm_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
	}
	else if (str_e==".png")
	{
		arma::Mat<unsigned char> MatriceChar(size(MatrixToSave,0),size(MatrixToSave,1));
		MatriceChar.ones();
	  
		status =MatriceChar.save(save_data, arma::raw_binary);
		ShowErrorMessageSaveWithLocation(save_data, status);
		
		
		
	}

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
  
  
  
std::string GetHomeDirectory()
{

	const char *homedir;

	if ((homedir = getenv("HOME")) == NULL) {
		homedir = getpwuid(getuid())->pw_dir;
	}

	std::string output=homedir;
  
  return output;
  
}
  
}
