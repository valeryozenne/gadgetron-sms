
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




void GenericReconSMSBase::apply_relative_phase_shift(hoNDArray< std::complex<float> >& data, bool is_positive )
{
    // à définir dans SMS Base car c'est aussi utilisé dans SMSPostGadget
    // vecteur_in_E1_direction=exp(1i*([1:1:size(reconSB,2)]- 1*center_k_space_sample   )*2*pi/PE_shift*(nband-1)) ;
    //  test=repmat( vecteur_in_E1_direction ,[ size(reconSB,1) 1 size(reconSB,3) size(reconSB,4) size(reconSB,5) size(reconSB,6)]  );
    //  reconSB(:,:,:,:,:,1,nt,nband)=reconSB(:,:,:,:,:,:,nt,nband).*test;

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    //show_size(data, "avant blip caipi");

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA, MB);

    center_k_space_E1=round(E1/2);

    GDEBUG_STREAM("  center_k_space_xml  "<<   center_k_space_xml  << " center_k_space_E1    "<<  center_k_space_E1<< " E1 "  << E1 );

    arma::fvec index_imag = arma::linspace<arma::fvec>( 0, E1-1, E1 )  - center_k_space_E1 ;

    //std::cout << index_imag << std::endl;

    arma::cx_fvec phase;
    arma::cx_fvec shift_to_apply;

    phase.set_size(E1);
    phase.zeros();
    phase.set_imag(index_imag);

    size_t m,a,n,s,cha,e2,e1,ro;
    float caipi_factor;

    int facteur;

    if (is_positive==true)
    {facteur=1;}
    else
    {facteur=-1;}

    for (m = 0; m < MB_factor; m++) {

        caipi_factor=2*arma::datum::pi/(facteur*Blipped_CAIPI)*(m);

        if (is_positive==true)
        {
            GDEBUG_STREAM(" apply Blipped_CAIPI  "<<   facteur*Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
        }
        else
        {
            GDEBUG_STREAM(" undo Blipped_CAIPI  "<<   facteur*Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
        }

        shift_to_apply=exp(phase*caipi_factor);

        // TODO in gadegtron 4.0 , try this:
        //for (e1 = 0; e1 < E1; e1++)
        //{
        //        phase_shift(slice,e1,slice,slice,slice)=shift_to_apply(e1);  // repmat variation only along e1 and mb once added TODO
        //}

        // TODO add mb dimension in phase_shift and then do the multiplication done
        // check how to use gprof
        for (cha = 0; cha < CHA; cha++)
        {
            for (e2 = 0; e2 < E2; e2++)
            {
                for (e1 = 0; e1 < E1; e1++)
                {
                    for (ro = 0; ro < RO; ro++)
                    {
                        phase_shift(ro,e1,e2,cha,m)=shift_to_apply(e1);  // repmat variation only along e1 and mb once added TODO
                    }
                }
            }
        }
    }

     data *= phase_shift;  //thansk to david

/*
        for (a = 0; a < lNumberOfStacks_; a++) {

            for (s = 0; s < S; s++)
            {
                size_t usedS = s;
                if (usedS >= S) usedS = S - 1;

                for (n = 0; n < N; n++)
                {
                    size_t usedN = n;
                    if (usedN >= N) usedN = N - 1;

                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                    Gadgetron::multiply(tempo, phase_shift, tempo);

                    memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                }
            }
        }
    }*/

}




void GenericReconSMSBase::apply_relative_phase_shift_test(hoNDArray< std::complex<float> >& data, bool is_positive )
{
    // à définir dans SMS Base car c'est aussi utilisé dans SMSPostGadget
    // vecteur_in_E1_direction=exp(1i*([1:1:size(reconSB,2)]- 1*center_k_space_sample   )*2*pi/PE_shift*(nband-1)) ;
    //  test=repmat( vecteur_in_E1_direction ,[ size(reconSB,1) 1 size(reconSB,3) size(reconSB,4) size(reconSB,5) size(reconSB,6)]  );
    //  reconSB(:,:,:,:,:,1,nt,nband)=reconSB(:,:,:,:,:,:,nt,nband).*test;

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    //show_size(data, "avant blip caipi");

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    center_k_space_E1=round(E1/2);

    //GDEBUG_STREAM("  center_k_space_xml  "<<   center_k_space_xml  << " center_k_space_E1    "<<  center_k_space_E1<< " E1 "  << E1 );

    arma::fvec index_imag = arma::linspace<arma::fvec>( 0, E1-1, E1 )  - center_k_space_E1 +2;

    //std::cout << index_imag << std::endl;

    arma::cx_fvec phase;
    arma::cx_fvec shift_to_apply;

    phase.set_size(E1);
    phase.zeros();
    phase.set_imag(index_imag);

    size_t m,a,n,s,cha,e2,e1,ro;
    float caipi_factor;

    int facteur;

    if (is_positive==true)
    {facteur=1;}
    else
    {facteur=-1;}


    for (m = 0; m < MB_factor; m++) {

        caipi_factor=2*arma::datum::pi/(facteur*Blipped_CAIPI)*(m);

        if (is_positive==true)
        {
            GDEBUG_STREAM(" apply Blipped_CAIPI  "<<   facteur*Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
        }
        else
        {
            GDEBUG_STREAM(" undo Blipped_CAIPI  "<<   facteur*Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
        }

        shift_to_apply=exp(phase*caipi_factor);

        for (e1 = 0; e1 < E1; e1++)
        {
            for (ro = 0; ro < RO; ro++)
            {
                for (e2 = 0; e2 < E2; e2++)
                {
                    for (cha = 0; cha < CHA; cha++)
                    {
                        phase_shift(ro,e1,e2,cha)=shift_to_apply(e1);
                    }
                }
            }
        }

        for (a = 0; a < lNumberOfStacks_; a++) {

            for (s = 0; s < S; s++)
            {
                size_t usedS = s;
                if (usedS >= S) usedS = S - 1;

                for (n = 0; n < N; n++)
                {
                    size_t usedN = n;
                    if (usedN >= N) usedN = N - 1;

                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                    Gadgetron::multiply(tempo, phase_shift, tempo);

                    memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                }
            }
        }
    }
}



void GenericReconSMSBase::get_header_and_position_and_gap(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > headers_)
{
    size_t E1 = data.get_size(1);
    size_t SLC=lNumberOfSlices_;
    size_t STK=lNumberOfStacks_;
    size_t MB=MB_factor;

    size_t e1,s,a,m;

    size_t start_E1(0), end_E1(0);
    auto t = Gadgetron::detect_sampled_region_E1(data);
    start_E1 = std::get<0>(t);
    end_E1 = std::get<1>(t);  

    hoNDArray<float > shift_from_isocenter;
    shift_from_isocenter.create(3);
    hoNDArray<float > read_dir;
    read_dir.create(3);
    hoNDArray<float > phase_dir;
    phase_dir.create(3);
    hoNDArray<float > slice_dir;
    slice_dir.create(3);

    arma::fvec z_offset(SLC);

    for (s = 0; s < SLC; s++)
    {
        ISMRMRD::AcquisitionHeader& curr_header = headers_(start_E1, 0, 0, 0, s);
        for (int j = 0; j < 3; j++) {

            shift_from_isocenter(j)=curr_header.position[j];
            read_dir(j)=curr_header.read_dir[j];
            phase_dir(j)=curr_header.phase_dir[j];
            slice_dir(j)=curr_header.slice_dir[j];
            //std::cout <<  curr_header.position[j]<< " "  <<  curr_header.read_dir[j] << " "  <<  curr_header.phase_dir[j]  << " "  <<  curr_header.slice_dir[j]  << std::endl;
        }

        z_offset(s) = dot(shift_from_isocenter,slice_dir);
    }

    //std::cout << z_offset <<std::endl;
    //std::cout << "   " <<z_offset.max()<< " " <<z_offset.min() << std::endl;

    //reorientation dans l'espace
    z_offset_geo.set_size(SLC);

    for (s = 0; s < SLC; s++)
    {
        z_offset_geo(s)=z_offset(indice_sb(s));
    }

    //std::cout << z_offset_geo << std::endl;

    arma::fvec delta_slice(SLC-1);

    for (s = 1; s < SLC; s++)
    {
        delta_slice(s-1)=z_offset_geo(s)-z_offset_geo(s-1);
    }

    //std::cout << delta_slice << std::endl;

    float  slice_gap_factor=(delta_slice(0)-slice_thickness)/slice_thickness*100;

    GDEBUG_STREAM("slice thickness is "<<  slice_thickness);
    GDEBUG_STREAM("slice distance is "<<  delta_slice(0));
    GDEBUG_STREAM("slice gap factor is "<<  slice_gap_factor<< " %" );

    //selection d'un jeux de données :
    arma::ivec index(MB);
    index.zeros();

    //index.print();

    z_gap.set_size(1);

    for (a = 0; a < 1; a++)
    {
        MapSliceSMS.row(a).print();
        index=MapSliceSMS.row(a).t();

        for (m = 0; m < MB-1; m++)
        {
            if (z_offset_geo(index(m+1))>z_offset_geo(index(m)))
            {
                GDEBUG_STREAM("distance au centre de la coupe la proche: " <<z_offset_geo(index(m))) ;
                GDEBUG_STREAM("distance entre les coupes simultanées: " <<  z_offset_geo(index(m+1))-z_offset_geo(index(m))) ;

                z_gap(m)=z_offset_geo(index(m+1))-z_offset_geo(index(m));
            }
        }
    }

    // std::cout << z_gap<< std::endl;
}

void GenericReconSMSBase::save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
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

        GERROR_STREAM(" save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim failed ... ");
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


void GenericReconSMSBase::save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number)
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
    else
    {
        //GDEBUG_STREAM(" saving 4D with STK_5 data soon ... ");
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

    corrneg_all_.set_size( dimensions_[0] , lNumberOfSlices_);
    corrpos_all_.set_size( dimensions_[0] , lNumberOfSlices_);
    corrneg_all_.zeros();
    corrpos_all_.zeros();

    corrneg_all_no_exp_.set_size( dimensions_[0] , lNumberOfSlices_);
    corrpos_all_no_exp_.set_size( dimensions_[0] , lNumberOfSlices_);
    corrneg_all_no_exp_.zeros();
    corrpos_all_no_exp_.zeros();

    for (s = 0; s < lNumberOfSlices_  ; s++)
    {
        GDEBUG_STREAM("Read EPI NAV from slice n° "<<  s);

        std::ostringstream oslc;
        oslc << s;

        arma::cx_fvec corrneg=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrneg_sms_slice_",   oslc.str(),  ".bin");
        arma::cx_fvec corrpos=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrpos_sms_slice_",   oslc.str(),  ".bin");

        arma::cx_fvec corrneg_no_exp=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrneg_sms_no_exp_slice_",   oslc.str(),  ".bin");
        arma::cx_fvec corrpos_no_exp=Gadgetron::LoadCplxVectorFromtheDisk("/tmp/", "gadgetron/", "corrpos_sms_no_exp_slice_",   oslc.str(),  ".bin");

        corrneg_all_.col(s)=corrneg;
        corrpos_all_.col(s)=corrpos;

        corrneg_all_no_exp_.col(s)=corrneg_no_exp;
        corrpos_all_no_exp_.col(s)=corrpos_no_exp;      

        GADGET_CHECK_THROW( size(corrneg,0) == dimensions_[0] );

        std::complex<float> * out_neg = &(epi_nav_neg_(0, s));
        memcpy(out_neg, &corrneg(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_pos = &(epi_nav_pos_(0, s));
        memcpy(out_pos, &corrpos(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_neg_no_exp = &(epi_nav_neg_no_exp_(0, s));
        memcpy(out_neg_no_exp, &corrneg_no_exp(0) , sizeof(std::complex<float>)*dimensions_[0]);

        std::complex<float> * out_pos_no_exp = &(epi_nav_pos_no_exp_(0, s));
        memcpy(out_pos_no_exp, &corrpos_no_exp(0) , sizeof(std::complex<float>)*dimensions_[0]);

        /*if (s==2)
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




void GenericReconSMSBase::compute_mean_epi_arma_nav(arma::cx_fcube &input,  arma::cx_fmat& output_no_exp,  arma::cx_fmat& output)
{

    size_t RO=size(input,0);
    size_t MB=size(input,1);
    size_t STK=size(input,2);

    for (size_t a = 0; a < STK; a++)
    {

        arma::cx_fmat nav=input.slice(a);

        arma::cx_fvec nav_sum_2nd=mean(nav,1);

        output_no_exp.col(a)=nav_sum_2nd;

        output.col(a)=arma::exp(nav_sum_2nd);    
    }
}



void GenericReconSMSBase::reorganize_arma_nav(arma::cx_fmat data, arma::uvec indice)
{
    size_t RO=size(data,0);
    size_t SLC=size(data,1);

    arma::cx_fmat new_data;
    new_data.set_size(RO, SLC);

    size_t n, s;

    for (int slc = 0; slc < SLC; slc++)
    {
        new_data.col(s)=data.col(indice(slc));
    }    

    data = new_data;

}


void GenericReconSMSBase::reorganize_nav(hoNDArray< std::complex<float> >& data, arma::uvec indice)
{
    size_t RO=data.get_size(0);
    size_t SLC=data.get_size(1);

    hoNDArray< std::complex<float> > new_data;
    new_data.create(RO, SLC);

    size_t n, s;

    for (int slc = 0; slc < SLC; slc++) {

        std::complex<float> * in = &(data(0, indice(slc)));
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



//sur les données single band
void GenericReconSMSBase::create_stacks_of_arma_nav(arma::cx_fmat &data, arma::cx_fcube &new_stack)
{
    size_t RO=size(data,0);
    size_t SLC=size(data,1);

    size_t MB=size(new_stack,1);
    size_t STK=size(new_stack,2);

    size_t a, m;
    size_t index;

    // copy of the data in the 8D array

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            new_stack.slice(a).col(m)=data.col(index);

            //std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
            //std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));
        }
    }
}


void GenericReconSMSBase::prepare_epi_data(size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    size_t RO=epi_nav_neg_.get_size(0);

    reorganize_nav(epi_nav_neg_, indice_sb);
    reorganize_nav(epi_nav_pos_, indice_sb);

    reorganize_arma_nav(corrneg_all_, indice_sb);
    reorganize_arma_nav(corrpos_all_, indice_sb);

    reorganize_nav(epi_nav_neg_no_exp_, indice_sb);
    reorganize_nav(epi_nav_pos_no_exp_, indice_sb);

    reorganize_arma_nav(corrneg_all_no_exp_, indice_sb);
    reorganize_arma_nav(corrpos_all_no_exp_, indice_sb);

    epi_nav_neg_STK_.create(RO, MB_factor, lNumberOfStacks_ );
    epi_nav_pos_STK_.create(RO, MB_factor, lNumberOfStacks_ );

    epi_nav_neg_no_exp_STK_.create(RO, MB_factor, lNumberOfStacks_ );
    epi_nav_pos_no_exp_STK_.create(RO, MB_factor, lNumberOfStacks_ );

    epi_nav_neg_no_exp_STK_mean_.create(RO, lNumberOfStacks_ );
    epi_nav_pos_no_exp_STK_mean_.create(RO, lNumberOfStacks_ );

    corrneg_all_STK_.set_size(RO, MB_factor, lNumberOfStacks_ );
    corrpos_all_STK_.set_size(RO, MB_factor, lNumberOfStacks_ );
    corrneg_all_STK_.zeros();
    corrpos_all_STK_.zeros();

    corrneg_all_no_exp_STK_.set_size(RO, MB_factor, lNumberOfStacks_ );
    corrpos_all_no_exp_STK_.set_size(RO, MB_factor, lNumberOfStacks_ );
    corrneg_all_no_exp_STK_.zeros();
    corrpos_all_no_exp_STK_.zeros();

    corrneg_all_no_exp_STK_mean_.set_size(RO, lNumberOfStacks_ );
    corrpos_all_no_exp_STK_mean_.set_size(RO, lNumberOfStacks_ );
    corrneg_all_no_exp_STK_mean_.zeros();
    corrpos_all_no_exp_STK_mean_.zeros();

    corrneg_all_STK_mean_.set_size(RO, lNumberOfStacks_ );
    corrpos_all_STK_mean_.set_size(RO, lNumberOfStacks_ );
    corrneg_all_STK_mean_.zeros();
    corrpos_all_STK_mean_.zeros();

    create_stacks_of_nav(epi_nav_neg_, epi_nav_neg_STK_);
    create_stacks_of_nav(epi_nav_pos_, epi_nav_pos_STK_);

    create_stacks_of_nav(epi_nav_neg_no_exp_, epi_nav_neg_no_exp_STK_);
    create_stacks_of_nav(epi_nav_pos_no_exp_, epi_nav_pos_no_exp_STK_);

    create_stacks_of_arma_nav(corrneg_all_, corrneg_all_STK_);
    create_stacks_of_arma_nav(corrpos_all_, corrpos_all_STK_);

    create_stacks_of_arma_nav(corrneg_all_no_exp_, corrneg_all_no_exp_STK_);
    create_stacks_of_arma_nav(corrpos_all_no_exp_, corrpos_all_no_exp_STK_);

    /*for (size_t a = 0; a < lNumberOfStacks_  ; a++)
    {
        for (size_t m = 0; m < MB_factor  ; m++)
        {
            std::cout << m  << " " << a << std::endl;
            std::cout <<  corrneg_all_STK_.slice(a).col(m)  << std::endl;
            std::cout <<  corrpos_all_STK_.slice(a).col(m)  << std::endl;
        }
    }*/

    compute_mean_epi_nav(epi_nav_neg_no_exp_STK_, epi_nav_neg_no_exp_STK_mean_);
    compute_mean_epi_nav(epi_nav_pos_no_exp_STK_, epi_nav_pos_no_exp_STK_mean_);

    compute_mean_epi_arma_nav(corrneg_all_no_exp_STK_, corrneg_all_no_exp_STK_mean_, corrneg_all_STK_mean_);
    compute_mean_epi_arma_nav(corrpos_all_no_exp_STK_, corrpos_all_no_exp_STK_mean_, corrpos_all_STK_mean_);

    /*for (size_t a = 0; a < lNumberOfStacks_  ; a++)
    {
        for (size_t ro = 0; ro < RO; ro++)
        {
        std::cout << " ro "<< corrneg_all_no_exp_STK_(ro, 0 , a) << "  "<< corrneg_all_no_exp_STK_mean_(ro, a) << "  " << corrneg_all_STK_(ro, 0 , a) << "  "<< corrneg_all_STK_mean_(ro, a) << "  " << std::endl;
        }
    }*/

    //gt_exporter_.export_array_complex(epi_nav_neg_no_exp_STK_, debug_folder_full_path_ + "NAV_neg_no_exp_STK" + os.str());
    //gt_exporter_.export_array_complex(epi_nav_pos_no_exp_STK_, debug_folder_full_path_ + "NAV_pos_no_exp_STK" + os.str());

    //save_4D_data(epi_nav_neg_no_exp_STK_, "NAV_neg_no_exp_STK", os.str());
    //save_4D_data(epi_nav_pos_no_exp_STK_, "NAV_pos_no_exp_STK", os.str());

    //gt_exporter_.export_array_complex(epi_nav_neg_no_exp_STK_mean_, debug_folder_full_path_ + "NAV_neg_no_exp_STK_mean" + os.str());
    //gt_exporter_.export_array_complex(epi_nav_pos_no_exp_STK_mean_, debug_folder_full_path_ + "NAV_pos_no_exp_STK_mean" + os.str());

    //save_4D_data(epi_nav_neg_no_exp_STK_mean_, "NAV_neg_no_exp_STK_mean", os.str());
    //save_4D_data(epi_nav_pos_no_exp_STK_mean_, "NAV_pos_no_exp_STK_mean", os.str());

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

    //GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_SB - end_E1_SB  : " << start_E1_SB << " - " << end_E1_SB);
    //GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_MB - end_E1_MB  : " << start_E1_MB << " - " << end_E1_MB);

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




bool GenericReconSMSBase::detect_first_repetition(IsmrmrdReconBit &recon_bit)
{
    bool is_first_repetition=true;

    for (size_t ii=0; ii<recon_bit.data_.headers_.get_number_of_elements(); ii++)
    {
        if( recon_bit.data_.headers_(ii).idx.repetition>0 )
        {
            GDEBUG_STREAM("GenericReconSMSBase - this is not the first repetition)");
            is_first_repetition=false;
            break;
        }
    }

    if (is_first_repetition) {GDEBUG_STREAM("GenericReconSMSBase - this is the first repetition)"); }

    return is_first_repetition;

}

bool GenericReconSMSBase::detect_single_band_data(IsmrmrdReconBit &recon_bit)
{
    bool is_single_band=false;

    for (size_t ii=0; ii<recon_bit.data_.headers_.get_number_of_elements(); ii++)
    {
        if( recon_bit.data_.headers_(ii).idx.user[0]==1 )
        {
            GDEBUG_STREAM("GenericReconSMSBase - Single band data (assuming splitSMSGadget is ON)");
            is_single_band=true;
            break;
        }
    }

    if (is_single_band==false) {GDEBUG_STREAM("GenericReconSMSBase - Multiband band data (assuming splitSMSGadget is ON)");}

    return is_single_band;

}

void GenericReconSMSBase::define_usefull_parameters_simple_version(IsmrmrdReconBit &recon_bit, size_t e)
{

    size_t start_E1_SB(0), end_E1_SB(0);

    //std::cout << "number of dimension "<< recon_bit.data_.data_.get_number_of_dimensions() << std::endl;

    auto t = Gadgetron::detect_sampled_region_E1(recon_bit.data_.data_);
    start_E1_SB = std::get<0>(t);
    end_E1_SB = std::get<1>(t);

    /*size_t start_E1_MB(0), end_E1_MB(0);
    t = Gadgetron::detect_sampled_region_E1(recon_bit.data_.data_);
    start_E1_MB = std::get<0>(t);
    end_E1_MB = std::get<1>(t);*/

    //GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - detect_sampled_region_E1 - start_E1_SB - end_E1_SB  : " << start_E1_SB << " - " << end_E1_SB);
    //GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - detect_sampled_region_E1 - start_E1_MB - end_E1_MB  : " << start_E1_MB << " - " << end_E1_MB);

    SamplingLimit sampling_limits_SB[3]; //, sampling_limits_MB[3];
    for (int i = 0; i < 3; i++)
        sampling_limits_SB[i] = recon_bit.data_.sampling_.sampling_limits_[i];

    //for (int i = 0; i < 3; i++)
    //   sampling_limits_MB[i] = recon_bit.data_.sampling_.sampling_limits_[i];

    /*for (int i = 0; i < 3; i++)
    {
        std::cout << "  sampling_limits_SB[i].min_ "<< sampling_limits_SB[i].min_ << "  sampling_limits_MB[i].min_ "<< sampling_limits_MB[i].min_ << std::endl;
        std::cout << "  sampling_limits_SB[i].center_ "<< sampling_limits_SB[i].center_ << "  sampling_limits_MB[i].center_ "<< sampling_limits_MB[i].center_ << std::endl;
        std::cout << "  sampling_limits_SB[i].max_ "<< sampling_limits_SB[i].max_ << "  sampling_limits_MB[i].max_ "<< sampling_limits_MB[i].max_ << std::endl;
    }*/

    /*for (int i = 0; i < 3; i++)
    {
        GADGET_CHECK_THROW(sampling_limits_SB[i].min_ <= sampling_limits_MB[i].min_);
        GADGET_CHECK_THROW(sampling_limits_SB[i].center_ <= sampling_limits_MB[i].center_);
        GADGET_CHECK_THROW(sampling_limits_SB[i].max_ <= sampling_limits_MB[i].max_);
    }*/

    //std::cout << start_E1_SB << "  "<< end_E1_SB << "  "<<  acceFactorSMSE1_[e] << std::endl;
    //std::cout << start_E1_MB << "  "<< end_E1_MB << "  "<<  acceFactorSMSE1_[e] << std::endl;

    size_t reduced_E1_SB_=get_reduced_E1_size(start_E1_SB,end_E1_SB, acceFactorSMSE1_[e] );
    //size_t reduced_E1_MB_=get_reduced_E1_size(start_E1_MB,end_E1_MB, acceFactorSMSE1_[e] );

    /*if (reduced_E1_SB_!= reduced_E1_MB_)
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
    }*/

    start_E1_=start_E1_SB;
    end_E1_=end_E1_SB;
    reduced_E1_=reduced_E1_SB_;

    //GDEBUG_STREAM("GenericReconCartesianSliceGrappaGadget - start_E1_ - end_E1_  : " << start_E1_ << " - " << end_E1_);

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


void GenericReconSMSBase::apply_ghost_correction_with_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal )
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

    size_t m, a, e1, ro, e2, cha, n, s;

    hoNDArray< std::complex<float> > shift;
    shift.create(RO);

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    hoNDFFT<float>::instance()->ifft(&data,0);

    // std::complex<float> signe(0,-1);
    // std::cout << " signe "<< signe <<std::endl;

    // on suppose que les corrections EPI sont les mêmes pour tous les S

    //std::cout << "start_E1_ "<<  start_E1_ <<  "end_E1_ "<<  end_E1_ <<   "acc "<<  acc << std::endl;

    for (m = 0; m < MB; m++) {

        for (a = 0; a < STK; a++) {

            //for (size_t e1 = start_E1_; e1 < end_E1_; e1+=acc)
            for (size_t e1 = start_E1_; e1 <= end_E1_; e1+=acc)
            {
                ISMRMRD::AcquisitionHeader& curr_header = headers_(e1, 0, 0, 0, 0);  //5D, fixed order [E1, E2, N, S, LOC]
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
                    //  if (a==0 && m==0 && ro==3 )
                    //  {std::cout << shift(ro) <<  "  " << exp(shift(ro)) <<"  "  << std::conj(shift(ro)) <<  "  " << exp(std::conj(shift(ro))) <<  std::endl;}

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

                    //if (a==0 && m==0 ) std::cout<< tempo(0, 0, 0, 0) << " " <<  phase_shift(0, 0, 0, 0) << std::endl;

                    Gadgetron::multiply(tempo, phase_shift, tempo);

                    memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                }
            }
        }
    }

    hoNDFFT<float>::instance()->fft(&data,0);
}





void GenericReconSMSBase::apply_ghost_correction_with_arma_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal , std::string msg)
{
    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t MB = data.get_size(4);
    size_t STK = data.get_size(5);
    size_t N = data.get_size(6);
    size_t S = data.get_size(7);

    //GDEBUG_STREAM("GenericReconSMSBase - EPI stk6 data array  : [RO E1 E2 CHA N S SLC] - [" << msg << " " << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N << " " << S << "]");

    size_t m, a, e1, ro, e2, cha, n, s;

    hoNDFFT<float>::instance()->ifft(&data,0);

    //size_t hE1 = headers_.get_size(0);
    //size_t hE2 = headers_.get_size(1);
    //size_t hN = headers_.get_size(2);
    //size_t hS = headers_.get_size(3);
    //size_t hSLC = headers_.get_size(4);

    //GDEBUG_STREAM("GenericReconSMSBase - EPI stk6 data array  HEAD : [E1 E2 N S SLC] - [" << msg << " " << hE1 << " " << hE2<< " " << hN << " " << hS <<" " << hSLC << "]");

    unsigned int compteur_pos;
    unsigned int compteur_neg;

    arma::cx_fvec tempo;
    tempo.set_size(RO);

    arma::cx_fvec correction_pos;
    correction_pos.set_size(RO);

    arma::cx_fvec correction_neg;
    correction_neg.set_size(RO);

     for (a = 0; a < STK; a++) {

          for (m = 0; m < MB; m++) {

            compteur_pos=0;
            compteur_neg=0;

            if (optimal==true)
            {
            correction_pos=corrpos_all_STK_.slice(a).col(m) ;
            correction_neg=corrneg_all_STK_.slice(a).col(m) ;
            }
            else
            {
            correction_pos=corrpos_all_STK_mean_.col(a) ;
            correction_neg=corrneg_all_STK_mean_.col(a) ;
            }

            if (undo==true)
            {
                correction_pos=corrpos_all_STK_.slice(a).col(m)/corrpos_all_STK_mean_.col(a) ;
                correction_neg=corrneg_all_STK_.slice(a).col(m)/corrneg_all_STK_mean_.col(a) ;
            }

            for (cha = 0; cha < CHA; cha++) {

                for (e2 = 0; e2 < E2; e2++)  {

                    for (size_t e1 = start_E1_; e1 <= end_E1_; e1+=acc)
                    {
                        ISMRMRD::AcquisitionHeader& curr_header = headers_(e1, 0, 0, 0, 0);  //5D, fixed order [E1, E2, N, S, LOC]
                        //std::cout << " e1 "  <<  e1 << "  ISMRMRD_ACQ_IS_REVERSE  " << curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)<< std::endl;

                        std::complex<float> * in = &(data(0, e1, e2, cha, m, a, n, s));
                        std::complex<float> * out = &(tempo(0));

                        memcpy(out , in, sizeof(std::complex<float>)*RO);

                        if (curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)) {
                            // Negative readout
                            compteur_neg++;

                            // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrneg_);
                            tempo %= correction_neg; // corrneg_mean_.col(get_stack_number_from_gt_numerotation(slice));

                            // Now that we have corrected we set the readout direction to positive
                        }
                        else {
                            // Positive readout
                            compteur_pos++;

                            // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrpos_);
                            tempo %=  correction_pos;  // corrpos_mean_.col(get_stack_number_from_gt_numerotation(slice));
                        }

                        std::complex<float> * in2 = &(tempo(0));
                        std::complex<float> * out2 = &(data(0, e1, e2, cha, m, a, n, s));
                        memcpy(out2 , in2, sizeof(std::complex<float>)*RO);

                    }
                }
            }

            if (compteur_pos==0)
            {GERROR_STREAM("apply_ghost_correction_with_arma_STK6 : compteur_pos is equal to 0  ... "<< compteur_pos<< " "<< compteur_neg);
            }
            if (compteur_neg==0)
            {GERROR_STREAM("apply_ghost_correction_with_arma_STK6 : compteur_pos is equal to 0  ... "<< compteur_pos<< " "<< compteur_neg); }


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

    //GADGET_CHECK_THROW(CHA==lNumberOfChannels_)
    GADGET_CHECK_THROW(STK==lNumberOfStacks_);

    //GDEBUG_STREAM("GenericSMSPrepGadget - mb stk7 data array  : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << STK  << "]");

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

        for (size_t e1 = start_E1_; e1 <= end_E1_; e1+=acc)
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






void GenericReconSMSBase::apply_absolute_phase_shift(hoNDArray< std::complex<float> >& data, bool is_positive)
{    

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    //GADGET_CHECK_THROW(CHA==lNumberOfChannels_)
    GADGET_CHECK_THROW(STK==lNumberOfStacks_);

    size_t m, a, n, s;
    size_t index;    

    std::complex<double> ii(0,1);

    int facteur;

    if (is_positive==true)
    {facteur=-1;}
    else
    {facteur=1;}

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            std::complex<double> lala=  exp(arma::datum::pi*facteur*ii*z_offset_geo(index)/z_gap(0));
            std::complex<float>  lili=  static_cast< std::complex<float> >(lala) ;           

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> *in = &(data(0, 0, 0, 0, m, a, n, s));

                    for (size_t j = 0; j < RO * E1 * E2 * CHA; j++) {

                        in[j] = in[j]*lili;
                    }
                }
            }
        }
    }
}

GADGET_FACTORY_DECLARE(GenericReconSMSBase)
}
