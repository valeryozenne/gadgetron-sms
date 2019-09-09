
/** \file   mri_core_multiband.cpp
    \brief  Implementation useful utility functionalities for thermometry
    \author Valery Ozenne
*/

#include "mri_core_multiband.h"
#include "mri_core_utility_interventional.h"
#include "hoNDArray_elemwise.h"
#include "hoArmadillo.h"
#include <pwd.h>

#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"

using namespace std;

namespace Gadgetron
{



arma::fvec get_information_from_multiband_special_card(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec output(13);
    output.zeros();


    ISMRMRD::TrajectoryDescription trajectory_description = *h.encoding[0].trajectoryDescription;

    if (h.encoding[0].trajectoryDescription)
    {
        for (std::vector<ISMRMRD::UserParameterDouble>::const_iterator i (trajectory_description.userParameterDouble.begin());
             i != trajectory_description.userParameterDouble.end(); i++)
        {
            if (i->name == "Unknow1") {
                output(0) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow1' found with value equal to %f \n", output(0) );
            }
            else if (i->name == "NMBSliceBands") {
                output(1) = i->value;
                //MB_factor_= i->value;
                GDEBUG(" Special Card Instructions 'NMBSliceBands' found with value equal to %f \n", output(1) );
            }
            else if (i->name == "Unknow2") {
                output(2) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow2' found with value equal to %f \n", output(2) );
            }
            else if (i->name == "MultiBandSliceInc") {
                output(3) = i->value;
                //MB_Slice_Inc_ = i->value;
                GDEBUG(" Special Card Instructions 'MultiBandSliceInc' found with value equal to %f \n", output(3) );
            }
            else if (i->name == "BlipFactorSL") {
                output(4) = i->value;
                //Blipped_CAIPI_ = i->value;
                GDEBUG(" Special Card Instructions 'BlipFactorSL' found with value equal to %f \n", output(4) );
            }
            else if (i->name == "Unknow3") {
                output(5) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow3' found with value equal to %f \n", output(5) );
            }
            else if (i->name == "Unknow4") {
                output(6) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow4' found with value equal to %f \n", output(6) );
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



arma::fvec get_information_from_wip_multiband_special_card(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec output(13);
    output.zeros();


    if (h.userParameters)
    {
        for (std::vector<ISMRMRD::UserParameterDouble>::const_iterator i (h.userParameters->userParameterDouble.begin());
                i != h.userParameters->userParameterDouble.end(); i++)
        {
            if (i->name == "Unknow1") {
                output(0) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow1' found with value equal to %f \n", output(0) );
            }
            else if (i->name == "MB_factor") {
                output(1) = i->value;
                //MB_factor_= i->value;
                GDEBUG(" Special Card Instructions 'NMBSliceBands' found with value equal to %f \n", output(1) );
            }
            else if (i->name == "Unknow2") {
                output(2) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow2' found with value equal to %f \n", output(2) );
            }
            else if (i->name == "MultiBandSliceInc") {
                output(3) = i->value;
                //MB_Slice_Inc_ = i->value;
                GDEBUG(" Special Card Instructions 'MultiBandSliceInc' found with value equal to %f \n", output(3) );
            }
            else if (i->name == "Blipped_CAIPI") {
                output(4) = i->value;
                //Blipped_CAIPI_ = i->value;
                GDEBUG(" Special Card Instructions 'BlipFactorSL' found with value equal to %f \n", output(4) );
            }
            else if (i->name == "Unknow3") {
                output(5) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow3' found with value equal to %f \n", output(5) );
            }
            else if (i->name == "Unknow4") {
                output(6) = i->value;
                GDEBUG(" Special Card Instructions 'Unknow4' found with value equal to %f \n", output(6) );
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

void save_kspace_data(arma::cx_fcube input, std::string str_home , string str_folder, std::string str_name)
{

    std::string str_e;
    std::string str_s;

    for (unsigned long s = 0; s < size(input,2); s++) {

        //ceci est le numero de slice ie de 0 à S-1
        std::ostringstream indice_slice;
        indice_slice << s;
        str_s = indice_slice.str();

        for (unsigned long e = 0; e < size(input,1); e++) {

            //ceci est le numero de dynamique ie de 0 à N-1
            std::ostringstream indice_encoding;
            indice_encoding << e;
            str_e = indice_encoding.str();

            Gadgetron::SaveVectorOntheDisk(input.slice(s).col(e),str_home, str_folder, str_name, str_e, str_s,  ".bin");
        }
    }
}



arma::ivec map_interleaved_acquisitions_mb(int number_of_stacks, bool no_reordering )
{

    arma::ivec index(number_of_stacks);
    index.zeros();


    if(no_reordering)
    {

        GDEBUG("CAUTION there is no reordering for multiband band images \n");

        for (unsigned int i = 0; i < number_of_stacks; i++)
        {
            index(i)=i;
        }
    }
    else
    {

        GDEBUG("Reordering with interleaved pattern for single band images \n");

        if (number_of_stacks%2)
        {
            index(0)=0;
            GDEBUG("Number of single band images is odd, incomprehensible rules are applied \n");
        }
        else
        {
            index(0)=0;
            GDEBUG("Number of single band images is even, interleaved rules are applied \n");
        }

        unsigned int compteur=0;

        for (unsigned int i = 1; i < number_of_stacks; i++)
        {

            if (number_of_stacks==18)
            {
                index(i)=index(i-1)+7;

                if (index(i)>=number_of_stacks)
                {
                    compteur++;
                    //std::cout << " i " << i <<  " index(i) "  << index(i)<< " compteur "   << compteur   << std::endl;

                    if (compteur==1 )
                    {
                        index(i)=3;
                    }
                    else if (compteur==2 )
                    {
                         index(i)=6;
                    }
                    else if (compteur==3 )
                    {
                         index(i)=2;
                    }
                    else if (compteur==4 )
                    {
                         index(i)=5;
                    }
                    else if (compteur==5 )
                    {
                         index(i)=1;
                    }
                    else if (compteur==6 )
                    {
                         index(i)=4;
                    }
                    else
                    {
                        index(i)=0;
                    }
                }
            }
            else if (number_of_stacks==12)
            {
                index(i)=index(i-1)+5;

                if (index(i)>=number_of_stacks)
                {
                    compteur++;
                    if (compteur==1 )
                    {
                        index(i)=3;
                     }
                    else if (compteur==2 )
                    {
                        index(i)=1;
                     }
                    else if (compteur==3 )
                    {
                        index(i)=4;
                    }
                    else if (compteur==4 )
                    {
                        index(i)=2;
                    }
                    else
                    {
                        index(i)=0;
                    }
                }
            }

            else if (number_of_stacks==10)
            {
                index(i)=index(i-1)+3;

                if (index(i)>=number_of_stacks)
                {
                    compteur++;
                    index(i)=3-compteur;
                }
            }
            else if (number_of_stacks==8)
            {

                index(i)=index(i-1)+3;

                if (index(i)>=number_of_stacks)
                {
                    compteur++;
                    index(i)=compteur;
                }

            }
            else if (number_of_stacks==6)
            {

                if (compteur==1)
                {
                index(i)=index(i-1)+4;
                }
                else
                {
                index(i)=index(i-1)+2;
                }

                if (index(i)>=number_of_stacks)
                {
                    compteur++;

                    if (compteur==1)
                    {
                        index(i)=1;
                    }
                    else if (compteur==2)
                    {
                        index(i)=3;
                    }
                    else
                    {
                        index(i)=0;
                    }
                }
            }
            else
            {

                index(i)=index(i-1)+2;

                if (index(i)>=number_of_stacks)
                {
                    if (number_of_stacks%2)
                    {
                        index(i)=1;
                    }
                    else
                    {
                        index(i)=1;
                    }
                }

            }

        }
    }

    //std::cout << "ordre d acquisition des coupes" << std::endl;
    //std::cout << index.t()  << std::endl;

    return index;
}


arma::ivec map_interleaved_acquisitions(int number_of_slices, bool no_reordering )
{

    arma::ivec index(number_of_slices);
    index.zeros();




    if(no_reordering)
    {
        GDEBUG("CAUTION there is no reordering for single band images \n");

        for (unsigned int i = 0; i < number_of_slices; i++)
        {
            index(i)=i;
        }
    }
    else
    {
        GDEBUG("Reordering with interleaved pattern for single band images \n");

        if (number_of_slices%2)
        {
            index(0)=0;
            GDEBUG("Number of single band images is odd \n");
        }
        else
        {
            index(0)=1;
            GDEBUG("Number of single band images is even \n");
        }

        for (unsigned int i = 1; i < number_of_slices; i++)
        {
            index(i)=index(i-1)+2;

            if (index(i)>=number_of_slices)
            {
                if (number_of_slices%2)
                {
                    index(i)=1;
                }
                else
                {
                    index(i)=0;
                }
            }

        }
    }    

    return index;
}

arma::imat get_map_slice_single_band(int MB_factor, int lNumberOfStacks, arma::ivec order_of_acquisition_mb, bool no_reordering)
{

    arma::imat output(MB_factor , lNumberOfStacks);
    output.zeros();

    if (lNumberOfStacks==1)
    {
        //std::cout << size(output)<< std::endl;
       // std::cout << size(map_interleaved_acquisitions(MB_factor, no_reordering ))<< std::endl;
        output=map_interleaved_acquisitions(MB_factor, no_reordering );
    }
    else
    {

        for (unsigned int a = 0; a < lNumberOfStacks; a++)    {

            //std::cout<<"  stack "   << a <<"  qui correspond à la coupe geometrique " << order_of_acquisition_mb(a)  << std::endl;

            //MapSliceSMS{i}= order_of_acquisition_mb(i):lNumberOfStacks_:number_of_slices-1;

            int count_map_slice=order_of_acquisition_mb(a);

            for (unsigned int m = 0; m < MB_factor; m++)
            {
                output(m,a) = count_map_slice;

                count_map_slice=count_map_slice+lNumberOfStacks;
            }

            //std::cout << a <<"  " << output.col(a).t()  << std::endl;
        }
    }

    return output;
}

}
