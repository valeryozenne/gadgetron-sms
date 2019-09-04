
#include "GenericReconSMSBase.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDFFT.h"

namespace Gadgetron {

GenericReconSMSBase::GenericReconSMSBase() : BaseClass()
{
}

GenericReconSMSBase::~GenericReconSMSBase()
{
}

int GenericReconSMSBase::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    ISMRMRD::IsmrmrdHeader h;
    try
    {
        deserialize(mb->rd_ptr(), h);
    }
    catch (...)
    {
        GDEBUG("Error parsing ISMRMRD Header");
    }



    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;

    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);




    /////////////////////////////////
    ///get_user_parameter_from_hdr
    /// assuming only one encoding

    std::vector<ISMRMRD::UserParameterDouble> vector_userParameterDouble;

    if (h.userParameters)
    {
        GDEBUG_STREAM("ParameterDouble in userParameters")
                vector_userParameterDouble =  h.userParameters->userParameterDouble;
    }
    else
    {
        GDEBUG_STREAM("ParameterDouble in trajectoryDescription")
                vector_userParameterDouble = h.encoding[0].trajectoryDescription->userParameterDouble;
    }

    is_wip_sequence=0;
    is_cmrr_sequence=0;

    /////////////////////////////////////
    ///find_acquisition_is_wip_or_cmrr
    ///

    //std::cout << vector_userParameterDouble.size() << std::endl;
    std::vector<ISMRMRD::UserParameterDouble>::const_iterator iter = vector_userParameterDouble.begin();

    for (; iter != vector_userParameterDouble.end(); iter++)
    {
        std::string usrParaName = iter->name;
        double usrParaValue = iter->value;

        //std::cout << " iter->name  "<<  iter->name << " iter->value "<< iter->value << std::endl;
        std::stringstream str;
        str << "MB_factor";

        std::stringstream str2;
        str2 << "NMBSliceBands";

        if (usrParaName == str.str() )
        {
            is_wip_sequence = 1;
            GDEBUG_STREAM("Sequence is Siemens WIP, find MB_factor : " << usrParaValue );
        }

        if (usrParaName == str2.str() )
        {
            is_cmrr_sequence = 1;
            GDEBUG_STREAM("Sequence is Siemens CMRR, find NMBSliceBands : " << usrParaValue );
        }
    }


    ////////////////////////////
    /// read MB_factor && Blipped_CAIPI parameters
    ///

    if (is_wip_sequence==1 && is_cmrr_sequence==0  )
    {
        iter = vector_userParameterDouble.begin();
        for (; iter != vector_userParameterDouble.end(); iter++)
        {
            std::string usrParaName = iter->name;
            double usrParaValue = iter->value;
            if (iter->name == "MB_factor") {
                MB_factor = iter->value;
            } else if (iter->name == "Blipped_CAIPI") {
                Blipped_CAIPI =iter->value;
            }
        }
    }
    else if (is_cmrr_sequence==1 && is_wip_sequence==0   )
    {

        iter = vector_userParameterDouble.begin();
        for (; iter != vector_userParameterDouble.end(); iter++)
        {
            std::string usrParaName = iter->name;
            double usrParaValue = iter->value;
            if (iter->name == "NMBSliceBands") {
                MB_factor = iter->value;
            } else if (iter->name == "BlipFactorSL") {
                Blipped_CAIPI =iter->value;
            }
        }
    }
    else
    {
        GERROR("is_wip_sequence && is_cmrr_sequence ");
    }

    GDEBUG_STREAM("Find MB factor : " << MB_factor << " find Blipped_CAIPI factor: "<< Blipped_CAIPI);

    ////////////////////////////
    /// adjust Blipped_CAIPI parameter if in plane acceleration
    ///

    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;
    GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

    acceFactorSMSE1_.resize(NE, 1);
    acceFactorSMSE2_.resize(NE, 1);
    //calib_SMSmode_.resize(NE, ISMRMRD_noacceleration);

    size_t e;
    for (e = 0; e < NE; e++)
    {
        if (!h.encoding[e].parallelImaging)
        {
            GDEBUG_STREAM("Parallel Imaging section not found in header for encoding space " << e);
            //calib_SMSmode_[e] = ISMRMRD_noacceleration;
            acceFactorSMSE1_[e] = 1;
            acceFactorSMSE2_[e] = 1;
        }
        else
        {
            ISMRMRD::ParallelImaging p_imaging = *h.encoding[e].parallelImaging;
            acceFactorSMSE1_[e] = p_imaging.accelerationFactor.kspace_encoding_step_1;
            acceFactorSMSE2_[e] = p_imaging.accelerationFactor.kspace_encoding_step_2;
            GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE1 is " << acceFactorSMSE1_[e]);
            GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE2 is " << acceFactorSMSE2_[e]);
        }
    }

    if (is_wip_sequence==1 && acceFactorSMSE1_[0]>1)
    {
        Blipped_CAIPI=Blipped_CAIPI*acceFactorSMSE1_[0];
        GDEBUG_STREAM("Find MB factor : " << MB_factor << " adjusted Blipped_CAIPI factor: "<< Blipped_CAIPI);
    }

    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;
    lNumberOfStacks_=lNumberOfSlices_/MB_factor;

    GDEBUG_STREAM(" lNumberOfSlices_  : " << lNumberOfSlices_ << " MB factor: "<< MB_factor << " lNumberOfStacks_: "<< lNumberOfStacks_ );


    //////////////
    /// \brief
    ///

    bool no_reordering=0;

    order_of_acquisition_mb=map_interleaved_acquisitions(lNumberOfStacks_, no_reordering);
    order_of_acquisition_sb=map_interleaved_acquisitions(lNumberOfSlices_, no_reordering);

    //std::cout <<  order_of_acquisition_mb << std::endl;
    //std::cout <<  order_of_acquisition_sb << std::endl;

    indice_mb =  arma::sort_index( order_of_acquisition_mb );
    indice_sb =  arma::sort_index( order_of_acquisition_sb );
    indice_slice_mb=indice_sb.rows(0,lNumberOfStacks_-1);

    // std::cout <<  indice_mb << std::endl;
    // std::cout <<  indice_sb << std::endl;
    // std::cout <<  indice_slice_mb << std::endl;

    // std::vector<hoNDArray<float> > MapSliceSMS;
    // MapSliceSMS.resize(lNumberOfStacks_);
    // for (size_t i = 0; i < MapSliceSMS.size(); ++i) {
    //    MapSliceSMS[i].create(MB_factor);
    // }


    //arma::uvec plot_mb= arma::sort( indice_sb );
    // std::cout << plot_mb << std::endl;  ;
    //for (unsigned int i = 0; i < lNumberOfStacks_; i++)
    //{
    //    std::cout << i <<   ;
    //}



    MapSliceSMS=get_map_slice_single_band( MB_factor,  lNumberOfStacks_,  order_of_acquisition_mb,  no_reordering);
    std::cout <<  MapSliceSMS<< std::endl;

    center_k_space_xml=h.encoding[0].encodingLimits.kspace_encoding_step_1->center+1;

    slice_thickness=h.encoding[0].encodedSpace.fieldOfView_mm.z;

    return GADGET_OK;
}

int GenericReconSMSBase::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{

    return GADGET_OK;
}




void GenericReconSMSBase::save_4D_data(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t X1 = input.get_size(0);
    size_t X2 = input.get_size(1);
    size_t X3 = input.get_size(2);
    size_t X4 = input.get_size(3);
    size_t X5 = input.get_size(4);
    size_t X6 = input.get_size(5);
    size_t X7 = input.get_size(6);

    if (  X5> 1 || X6> 1 || X7 >1 )
    {

        GERROR_STREAM(" save_4D_data failed ... ");
    }

    hoNDArray< std::complex<float> > output;
    output.create(X1, X2, X3 , X4);

    memcpy(&output(0, 0, 0, 0), &input(0, 0, 0, 0), sizeof(std::complex<float>)*X1*X2*X3*X4);

    if (!debug_folder_full_path_.empty())
    {
        gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number);
    }

    output.clear();
}



void GenericReconSMSBase::save_4D_with_SLC_7(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t N = input.get_size(4);
    size_t S = input.get_size(5);
    size_t SLC = input.get_size(6);

    if ( N> 1 || S> 1  )
    {

        GERROR_STREAM(" save_4D_with_SLC failed ... ");
    }

    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, CHA , SLC);

    size_t slc;

    for (slc = 0; slc < SLC; slc++)
    {
        memcpy(&output(0, 0, 0, slc), &input(0, 0, 0, 0 ,0, 0, slc), sizeof(std::complex<float>)*RO*E1*CHA);
    }

    if (!debug_folder_full_path_.empty())
    {
        gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number);
    }

    output.clear();
}


void GenericReconSMSBase::save_4D_with_STK_8(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, CHA , MB);

    size_t a,m;

    for (a = 0; a < STK; a++)
    {
        std::stringstream stk;
        stk << "_stack_" << a;

        for (m = 0; m < MB; m++)
        {
            memcpy(&output(0, 0, 0, m), &input(0, 0, 0, 0 ,m, a, 0, 0), sizeof(std::complex<float>)*RO*E1*CHA);
        }

        if (!debug_folder_full_path_.empty())
        {
            gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number + stk.str());
        }

    }



    output.clear();
}


void GenericReconSMSBase::save_4D_with_STK_5(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t CHA = input.get_size(2);
    size_t MB = input.get_size(3);
    size_t STK = input.get_size(4);
    size_t X6 = input.get_size(5);
    size_t X7 = input.get_size(6);

    if ( X6> 1 || X7> 1  )
    {
        GERROR_STREAM(" save_4D_5D_data failed ... ");
    }



    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, CHA , MB);

    size_t a;

    for (a = 0; a < STK; a++)
    {
        std::stringstream stk;
        stk << "_stack_" << a;

        memcpy(&output(0, 0, 0, 0), &input(0, 0, 0, 0 ,a), sizeof(std::complex<float>)*RO*E1*CHA*MB);

        gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number + stk.str());

    }

    output.clear();
}



void GenericReconSMSBase::save_4D_with_STK_6(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t CHA = input.get_size(2);
    size_t MB = input.get_size(3);
    size_t N  = input.get_size(4);
    size_t STK = input.get_size(5);

    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, CHA , MB);

    size_t a;

    for (a = 0; a < STK; a++)
    {
        std::stringstream stk;
        stk << "_stack_" << a;

        memcpy(&output(0, 0, 0, 0), &input(0, 0, 0, 0 , 0 , a), sizeof(std::complex<float>)*RO*E1*CHA*MB);

        gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number + stk.str());
    }

    output.clear();
}

void GenericReconSMSBase::save_4D_with_STK_7(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t CHA = input.get_size(2);
    size_t MB = input.get_size(3);
    size_t STK = input.get_size(4);
    size_t N = input.get_size(5);
    size_t S = input.get_size(6);

    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, CHA , MB);

    size_t a,m;

    for (a = 0; a < STK; a++)
    {
        std::stringstream stk;
        stk << "_stack_" << a;

        for (m = 0; m < MB; m++)
        {

            memcpy(&output(0, 0, 0, m), &input(0, 0, 0, 0 ,m, a, 0, 0), sizeof(std::complex<float>)*RO*E1*CHA*MB);

        }

        if (!debug_folder_full_path_.empty())
        {
            gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number + stk.str());
        }
    }

    output.clear();
}


void GenericReconSMSBase::save_4D_8D_kspace(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    hoNDArray< std::complex<float> > output;
    output.create(RO, E1, E2 , CHA);

    size_t a,m;


    for (a = 0; a < 1; a++)
    {
        //std::stringstream stk;
        //stk << "_stack_" << a;

        for (m = 0; m < MB; m++)
        {

            std::stringstream mm;
            mm << "_mb_" << m;

            memcpy(&output(0, 0, 0, 0), &input(0, 0, 0, 0 ,m, a, 0, 0), sizeof(std::complex<float>)*RO*E1*E2*CHA);

            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(output, debug_folder_full_path_ + name + encoding_number + mm.str());
            }

        }
    }

    output.clear();
}



arma::ivec GenericReconSMSBase::map_interleaved_acquisitions(int number_of_slices, bool no_reordering )
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


arma::imat GenericReconSMSBase::get_map_slice_single_band(int MB_factor, int lNumberOfStacks, arma::ivec order_of_acquisition_mb, bool no_reordering)
{
    arma::imat output(lNumberOfStacks, MB_factor);
    output.zeros();

    if (lNumberOfStacks==1)
    {
        output=map_interleaved_acquisitions(MB_factor, no_reordering );
    }
    else
    {
        for (unsigned int a = 0; a < lNumberOfStacks; a++)
        {
            int count_map_slice=order_of_acquisition_mb(a);

            for (unsigned int m = 0; m < MB_factor; m++)
            {
                output(a,m) = count_map_slice;
                count_map_slice=count_map_slice+lNumberOfStacks;
            }
        }
    }
    return output;
}



void GenericReconSMSBase::show_size(hoNDArray< std::complex<float> >& input, std::string name)
{
    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t X5 = input.get_size(4);
    size_t X6 = input.get_size(5);
    size_t X7 = input.get_size(6);
    size_t X8 = input.get_size(7);

    GDEBUG_STREAM("GenericReconSMSBase - "<<  name << ": [X1 X2 X3 X4 X5 X6 X7 X8 ] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << X5 <<  " " << X6 << " " << X7 << " " << X8  << "]");
}



void GenericReconSMSBase::load_epi_data()
{
    ///------------------------------------------------------------------------
    /// FR relecture des corrections single band EPI
    /// UK

    size_t s;

    epi_nav_neg_.create(dimensions_[0], lNumberOfSlices_);
    epi_nav_pos_.create(dimensions_[0], lNumberOfSlices_);

    epi_nav_neg_no_exp_.create(dimensions_[0], lNumberOfSlices_);
    epi_nav_pos_no_exp_.create(dimensions_[0], lNumberOfSlices_);

    for (s = 0; s < lNumberOfSlices_  ; s++)
    {
        GDEBUG_STREAM("Read EPI NAV from slice n° "<<  s);

        std::ostringstream oslc;
        oslc << s;

        arma::cx_fvec corrneg=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrneg_",   oslc.str(),  ".bin");
        arma::cx_fvec corrpos=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrpos_",   oslc.str(),  ".bin");

        arma::cx_fvec corrneg_no_exp=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrneg_no_exp_",   oslc.str(),  ".bin");
        arma::cx_fvec corrpos_no_exp=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrpos_no_exp_",   oslc.str(),  ".bin");

        /*if (s==2)
        {
            std::cout << exp(corrneg_no_exp)<< std::endl;
            //std::cout << exp(corrpos_no_exp)<< std::endl;
        }*/

        GADGET_CHECK_THROW( size(corrneg,0) == dimensions_[0] );

        std::complex<float> * out_neg = &(epi_nav_neg_(0, s));
        memcpy(out_neg, &corrneg(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_pos = &(epi_nav_pos_(0, s));
        memcpy(out_pos, &corrpos(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_neg_no_exp = &(epi_nav_neg_no_exp_(0, s));
        memcpy(out_neg_no_exp, &corrneg_no_exp(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_pos_no_exp = &(epi_nav_pos_no_exp_(0, s));
        memcpy(out_pos_no_exp, &corrpos_no_exp(0) , sizeof(std::complex<float>)*dimensions_[0]);

        /*
        if (s==2)
        {
            for (size_t ro = 0; ro < dimensions_[0]; ro++)
            {
                std::cout << " ro "<< epi_nav_neg_(ro,s) << " "<< epi_nav_neg_no_exp_(ro, s)  << std::endl;
            }
        }*/
    }
}



void GenericReconSMSBase::compute_mean_epi_nav(hoNDArray< std::complex<float> >& input,  hoNDArray< std::complex<float> >& output)
{

    size_t RO=input.get_size(0);
    size_t MB=input.get_size(1);
    size_t STK=input.get_size(2);

    for (size_t a = 0; a < STK; a++)
    {
        hoNDArray<std::complex<float> > nav(RO, MB);
        hoNDArray<std::complex<float> > nav_sum_2nd(RO,1);

        std::complex<float> * in = &(input(0, 0, a));
        std::complex<float> * out = &(nav(0, 0 ));

        memcpy(out, in, sizeof(std::complex<float>)*RO*MB);

        Gadgetron::sum_over_dimension(nav, nav_sum_2nd, 1);
        Gadgetron::scal( (1.0/MB), nav_sum_2nd);

        std::complex<float> * in2 = &(nav_sum_2nd(0));
        std::complex<float> * out2 = &(output(0, a ));

        memcpy(out2, in2, sizeof(std::complex<float>)*RO);
    }
}


void GenericReconSMSBase::reorganize_nav(hoNDArray< std::complex<float> >& data, arma::uvec indice)
{
    size_t RO=data.get_size(0);
    size_t SLC=data.get_size(1);

    hoNDArray< std::complex<float> > new_data;
    new_data.create(RO, SLC);

    size_t n, s;

    for (int slc = 0; slc < SLC; slc++) {

        std::complex<float> * in = &(data(0, indice[slc]));
        std::complex<float> * out = &(new_data(0, slc));

        memcpy(out , in, sizeof(std::complex<float>)*RO);
    }

    data = new_data;

}

//sur les données single band
void GenericReconSMSBase::create_stacks_of_nav(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack)
{
    size_t RO=data.get_size(0);
    size_t SLC=data.get_size(1);

    size_t MB=new_stack.get_size(1);
    size_t STK=new_stack.get_size(2);

    size_t a, m;
    size_t index;

    // copy of the data in the 8D array

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            std::complex<float> * in = &(data(0, index));
            std::complex<float> * out = &(new_stack(0, m, a));
            memcpy(out , in, sizeof(std::complex<float>)*RO);
        }
    }
}




void GenericReconSMSBase::prepare_epi_data()
{
    size_t RO=epi_nav_neg_.get_size(0);

    reorganize_nav(epi_nav_neg_, indice_sb);
    reorganize_nav(epi_nav_pos_, indice_sb);

    reorganize_nav(epi_nav_neg_no_exp_, indice_sb);
    reorganize_nav(epi_nav_pos_no_exp_, indice_sb);

    epi_nav_neg_STK_.create(RO, MB_factor, lNumberOfStacks_ );
    epi_nav_pos_STK_.create(RO, MB_factor, lNumberOfStacks_ );

    create_stacks_of_nav(epi_nav_neg_, epi_nav_neg_STK_);
    create_stacks_of_nav(epi_nav_pos_, epi_nav_pos_STK_);

    epi_nav_neg_no_exp_STK_.create(RO, MB_factor, lNumberOfStacks_ );
    epi_nav_pos_no_exp_STK_.create(RO, MB_factor, lNumberOfStacks_ );

    create_stacks_of_nav(epi_nav_neg_no_exp_, epi_nav_neg_no_exp_STK_);
    create_stacks_of_nav(epi_nav_pos_no_exp_, epi_nav_pos_no_exp_STK_);

    epi_nav_neg_no_exp_STK_mean_.create(RO, lNumberOfStacks_ );
    epi_nav_pos_no_exp_STK_mean_.create(RO, lNumberOfStacks_ );

    compute_mean_epi_nav(epi_nav_neg_no_exp_STK_, epi_nav_neg_no_exp_STK_mean_);
    compute_mean_epi_nav(epi_nav_pos_no_exp_STK_, epi_nav_pos_no_exp_STK_mean_);

 /*   for (size_t a = 0; a < lNumberOfStacks_; a++)
    {
        for (size_t m = 0; m < MB_factor; m++)
        {
            std::cout << " a" << a <<  " m " << m <<"  " <<epi_nav_neg_no_exp_STK_(0,m,a) << std::endl;
        }

        std::cout  << " a" << a <<  epi_nav_neg_no_exp_STK_mean_(0,a) << std::endl;
    }


    size_t a = 0;
    size_t m = 0;

    for (size_t ro = 0; ro < dimensions_[0]; ro++)
    {
        std::cout << " ro "<< epi_nav_neg_no_exp_STK_mean_(ro,a) << "   "<< epi_nav_pos_no_exp_STK_mean_(ro,a)  << std::endl;
    }
    std::cout << " -----------------"<<  std::endl;

    for (size_t ro = 0; ro < dimensions_[0]; ro++)
    {
        std::cout << " ro "<< epi_nav_neg_no_exp_STK_(ro,m,a) << "   "<< epi_nav_pos_no_exp_STK_(ro,m,a)  << std::endl;
    }
*/

}



void GenericReconSMSBase::define_usefull_parameters(IsmrmrdReconBit &recon_bit, size_t e)
{

    size_t start_E1_SB(0), end_E1_SB(0);
    auto t = Gadgetron::detect_sampled_region_E1(recon_bit.sb_->data_);
    start_E1_SB = std::get<0>(t);
    end_E1_SB = std::get<1>(t);

    size_t start_E1_MB(0), end_E1_MB(0);
    t = Gadgetron::detect_sampled_region_E1(recon_bit.data_.data_);
    start_E1_MB = std::get<0>(t);
    end_E1_MB = std::get<1>(t);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_SB - end_E1_SB  : " << start_E1_SB << " - " << end_E1_SB);
    GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_MB - end_E1_MB  : " << start_E1_MB << " - " << end_E1_MB);

    SamplingLimit sampling_limits_SB[3], sampling_limits_MB[3];
    for (int i = 0; i < 3; i++)
        sampling_limits_SB[i] = recon_bit.sb_->sampling_.sampling_limits_[i];

    for (int i = 0; i < 3; i++)
        sampling_limits_MB[i] = recon_bit.data_.sampling_.sampling_limits_[i];

    /*for (int i = 0; i < 3; i++)
    {
        std::cout << "  sampling_limits_SB[i].min_ "<< sampling_limits_SB[i].min_ << "  sampling_limits_MB[i].min_ "<< sampling_limits_MB[i].min_ << std::endl;
        std::cout << "  sampling_limits_SB[i].center_ "<< sampling_limits_SB[i].center_ << "  sampling_limits_MB[i].center_ "<< sampling_limits_MB[i].center_ << std::endl;
        std::cout << "  sampling_limits_SB[i].max_ "<< sampling_limits_SB[i].max_ << "  sampling_limits_MB[i].max_ "<< sampling_limits_MB[i].max_ << std::endl;
    }*/

    for (int i = 0; i < 3; i++)
    {
        GADGET_CHECK_THROW(sampling_limits_SB[i].min_ <= sampling_limits_MB[i].min_);
        GADGET_CHECK_THROW(sampling_limits_SB[i].center_ <= sampling_limits_MB[i].center_);
        GADGET_CHECK_THROW(sampling_limits_SB[i].max_ <= sampling_limits_MB[i].max_);
    }

    size_t reduced_E1_SB_=get_reduced_E1_size(start_E1_SB,end_E1_SB, acceFactorSMSE1_[e] );
    size_t reduced_E1_MB_=get_reduced_E1_size(start_E1_MB,end_E1_MB, acceFactorSMSE1_[e] );

    if (reduced_E1_SB_!= reduced_E1_MB_)
    {
        // on prend les dimensions les plus petites
        if (reduced_E1_SB_<reduced_E1_MB_)
        {
            start_E1_=start_E1_SB;
            end_E1_=end_E1_SB;
            reduced_E1_=reduced_E1_SB_;
        }
        else if (reduced_E1_SB_>reduced_E1_MB_)
        {
            start_E1_=start_E1_MB;
            end_E1_=end_E1_MB;
            reduced_E1_=reduced_E1_MB_;
        }
    }
    else
    {
        if (start_E1_SB!=start_E1_MB  || end_E1_SB!=end_E1_MB  )
        {
            GERROR("start_E1_SB!=start_E1_MB  || end_E1_SB!=end_E1_MB\n");
        }

        start_E1_=start_E1_SB;
        end_E1_=end_E1_SB;
        reduced_E1_=reduced_E1_SB_;
    }

    GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_ - end_E1_  : " << start_E1_ << " - " << end_E1_);

}


int GenericReconSMSBase::get_reduced_E1_size(size_t start_E1 , size_t end_E1 , size_t acc )
{
    int output;

    if (acc <= 1)  {
        output=(int)(end_E1-start_E1)+1;
    } else {
        output=(int)round((float)(end_E1-start_E1)/(float)acc)+1;
    }

    return output;
}


void GenericReconSMSBase::apply_ghost_correction_with_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool optimal )
{
    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t MB = data.get_size(4);
    size_t STK = data.get_size(5);
    size_t N = data.get_size(6);
    size_t S = data.get_size(7);

    GDEBUG_STREAM("GenericSMSPrepGadget - sb stk6 apply data array  : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N << " " << S << "]");

    size_t m,a, e1, ro, e2, cha, n, s;

    hoNDArray< std::complex<float> > shift;
    shift.create(RO);

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    hoNDFFT<float>::instance()->ifft(&data,0);

    // on suppose que les corrections EPI sont les mêmes pour tous les N et S

    for (m = 0; m < MB; m++) {

        for (a = 0; a < STK; a++) {

            //for (size_t e1 = start_E1_; e1 < end_E1_; e1+=acc)
            for (size_t e1 = start_E1_; e1 < end_E1_; e1+=acc)
            {
                ISMRMRD::AcquisitionHeader& curr_header = headers_(e1, 0, 0, 0, 0);
                //std::cout << "  ISMRMRD_ACQ_IS_REVERSE  " << curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)<< std::endl;

                if (optimal)
                {
                    if (curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)==1)
                    {
                        for (ro = 0; ro < RO; ro++)
                        {
                            shift(ro)=epi_nav_neg_no_exp_STK_(ro,m,a);
                        }
                    } else {
                        for (ro = 0; ro < RO; ro++)
                        {
                            shift(ro)=epi_nav_pos_no_exp_STK_(ro,m,a);
                        }
                    }
                }
                else
                {
                    if (curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)==1)
                    {
                        for (ro = 0; ro < RO; ro++)
                        {
                            shift(ro)=epi_nav_neg_no_exp_STK_mean_(ro,a);
                        }

                    } else {

                        for (ro = 0; ro < RO; ro++)
                        {
                            shift(ro)=epi_nav_pos_no_exp_STK_mean_(ro,a);
                        }
                    }
                }

                for (ro = 0; ro < RO; ro++)
                {
                    for (e2 = 0; e2 < E2; e2++)
                    {
                        for (cha = 0; cha < CHA; cha++)
                        {
                            phase_shift(ro,e1,e2,cha)=exp(shift(ro));
                        }
                    }
                }

            }

            for (n = 0; n < N; n++) {

                for (s = 0; s < S; s++) {

                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                    Gadgetron::multiply(tempo, phase_shift, tempo);

                    memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                }
            }
        }
    }

    hoNDFFT<float>::instance()->fft(&data,0);
}





void GenericReconSMSBase::apply_ghost_correction_with_STK7(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool optimal )
{
    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t N = data.get_size(4);
    size_t S = data.get_size(5);
    size_t STK = data.get_size(6);

    GDEBUG_STREAM("GenericSMSPrepGadget - mb stk7 data array  : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << STK  << "]");

    size_t a, e1, ro, e2, cha, n, s;

    hoNDArray< std::complex<float> > shift;
    shift.create(RO);

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    hoNDFFT<float>::instance()->ifft(&data,0);

    // on suppose que les corrections EPI sont les mêmes pour tous les N et S

    for (a = 0; a < STK; a++) {

        for (size_t e1 = start_E1_; e1 < end_E1_; e1+=acc)
        {
            ISMRMRD::AcquisitionHeader& curr_header = headers_(e1, 0, 0, 0, 0);
            //std::cout << "  ISMRMRD_ACQ_IS_REVERSE  " << curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)<< std::endl;

            if (optimal)
            {

            }
            else
            {
                if (curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)==1)
                {
                    for (ro = 0; ro < RO; ro++)
                    {
                        shift(ro)=epi_nav_neg_no_exp_STK_mean_(ro,a);
                    }

                } else {

                    for (ro = 0; ro < RO; ro++)
                    {
                        shift(ro)=epi_nav_pos_no_exp_STK_mean_(ro,a);
                    }
                }
            }

            for (ro = 0; ro < RO; ro++)
            {
                for (e2 = 0; e2 < E2; e2++)
                {
                    for (cha = 0; cha < CHA; cha++)
                    {
                        phase_shift(ro,e1,e2,cha)=exp(shift(ro));
                    }
                }
            }
        }

        for (n = 0; n < N; n++) {

            for (s = 0; s < S; s++) {

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, a));
                std::complex<float> * out = &(tempo(0, 0, 0, 0));
                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                Gadgetron::multiply(tempo, phase_shift, tempo);
                memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);
            }
        }
    }


    hoNDFFT<float>::instance()->fft(&data,0);
}




GADGET_FACTORY_DECLARE(GenericReconSMSBase)
}
