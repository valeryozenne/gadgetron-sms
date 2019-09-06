
#include "GenericReconSMSPrepv0Gadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSMSPrepv0Gadget::GenericReconSMSPrepv0Gadget() : BaseClass()
{
}

GenericReconSMSPrepv0Gadget::~GenericReconSMSPrepv0Gadget()
{
}

int GenericReconSMSPrepv0Gadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);




    return GADGET_OK;
}

int GenericReconSMSPrepv0Gadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSMSPrepv0Gadget::process"); }

    process_called_times_++;

    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
    if (recon_bit_->rbit_.size() > num_encoding_spaces_)
    {
        GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
    }

    // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
    size_t e, n, s, slc;
    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;


        if (recon_bit_->rbit_[e].sb_  &&  recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            define_usefull_parameters(recon_bit_->rbit_[e], e);
        }


        if (recon_bit_->rbit_[e].ref_)
        {
            // std::cout << " je suis la structure qui contient les données acs" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].ref_->data_;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");
        }

        if (recon_bit_->rbit_[e].sb_)
        {
            // std::cout << " je suis la structure qui contient les données single band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].sb_->data_;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array sb : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            save_4D_with_SLC_7(data, "FID_SB4D", os.str());

            permute_slices(data, indice_sb);

            save_4D_with_SLC_7(data, "FID_SB4D_permute_slices", os.str());

            hoNDArray< std::complex<float> > data8D;
            data8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , N, S );
            create_stacks_of_slices(data, data8D);

            save_4D_with_STK_8(data8D, "FID_SB4D_create", os.str());

            if (is_wip_sequence==1)
            {
                // si WIP on applique le blip caipi
                apply_relative_phase_shift(data8D);
                save_4D_with_STK_8(data8D, "FID_SB4D_relative_shift", os.str());

                // et on applique aussi l'offset de phase
                // recupération de l'offset de position dans la direction de coupe
                hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit_->rbit_[e].sb_->headers_;
                get_header_and_position_and_gap(data, headers_,  E1);

                apply_absolute_phase_shift(data8D);
                save_4D_with_STK_8(data8D, "FID_SB4D_absolute_shift", os.str());

            }
            else if (is_cmrr_sequence==1 && is_wip_sequence==0)
            {
                // si CMMR on ne fait rien
            }
            else
            {
                GERROR("is_wip_sequence && is_cmrr_sequence");
            }

            //apply the average slice navigator

            load_epi_data();

            //prepare epi data

            prepare_epi_data();

            save_4D_with_STK_8(data8D, "FID_SB_avant_epi_nav", os.str());

            apply_ghost_correction_with_STK6(data8D, recon_bit_->rbit_[e].sb_->headers_ ,  acceFactorSMSE1_[e] , false);

            save_4D_with_STK_8(data8D, "FID_SB_apres_epi_nav", os.str());

            m1->getObjectPtr()->rbit_[e].sb_->data_ = data8D;

        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            // std::cout << " je suis la structure qui contient les données multi band" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array data: [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            if (!debug_folder_full_path_.empty())
            {
                gt_exporter_.export_array_complex(data, debug_folder_full_path_ + "mb" + os.str());
            }

            save_4D_with_SLC_7(data, "FID_MB4D", os.str());

            remove_extra_dimension_and_permute_stack_dimension(data);

            save_4D_with_SLC_7(data, "FID_MB4D_remove", os.str());

            size_t STK = data.get_size(6);

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array data: [RO E1 E2 CHA N S STK ] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N <<  " " << S << " " << STK  << "]");

            // code usefull only for matlab comparison
            // reorganize_data(data, indice_mb);
            // save_4D_data(data, "FID_MB4D_reorganize", os.str());
            // reorganize_data(data, arma::conv_to<arma::uvec>::from(order_of_acquisition_mb));

            show_size(data,"FID_MB4D_reorganize_again" );
            save_4D_with_SLC_7(data, "FID_MB4D_reorganize_again", os.str());

            //apply the average slice navigator

            save_4D_with_SLC_7(data, "FID_MB_avant_epi_nav", os.str());

            apply_ghost_correction_with_STK7(data, recon_bit_->rbit_[e].data_.headers_ ,  acceFactorSMSE1_[e] , false);

            save_4D_with_SLC_7(data, "FID_MB_apres_epi_nav", os.str());

            //m1->getObjectPtr()->rbit_[e].data_.data_ = data;

        }
    }

    if (perform_timing.value()) { gt_timer_.stop(); }

    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}




void GenericReconSMSPrepv0Gadget::get_header_and_position_and_gap(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > headers_, size_t E1)
{
    size_t SLC=lNumberOfSlices_;
    size_t STK=lNumberOfStacks_;
    size_t MB=MB_factor;

    size_t e1,s,a,m;

    size_t start_E1(0), end_E1(0);
    auto t = Gadgetron::detect_sampled_region_E1(data);
    start_E1 = std::get<0>(t);
    end_E1 = std::get<1>(t);

    //std::cout <<  " start_E1 " <<  start_E1 << "  end_E1  "    << end_E1 << std::endl;

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

    z_gap.set_size(MB);
    z_gap.zeros();

    //std::cout << size(MapSliceSMS,0) <<  " "<<  size(MapSliceSMS,1)<<  std::endl;
    //MapSliceSMS.print();

    for (a = 0; a < 1; a++)
    {

        std::cout << size(index,0) <<  " "<<  size(index,1)<< "  "<< size(MapSliceSMS.row(a))  << std::endl;
        index=MapSliceSMS.row(a).t();

        //index.print();
        //z_offset_geo.print();

        for (m = 0; m < MB-1; m++)
        {           

            std::cout << index(m) <<"  " <<  index(m+1)<< std::endl;

            if (z_offset_geo(index(m+1))>z_offset_geo(index(m)))
            {
                // GDEBUG_STREAM("distance au centre de la coupe la proche: " <<z_offset_geo(index(m))) );
                // GDEBUG_STREAM("distance entre les coupes simultanées: " <<  z_offset_geo(index(m+1))-z_offset_geo(index(m))) );

                z_gap(m)=z_offset_geo(index(m+1))-z_offset_geo(index(m));
            }

        }
    }
//TODO Erreur de segmentation (core dumped) si c'est decommenté

     //std::cout << z_gap<< std::endl;
}


void GenericReconSMSPrepv0Gadget::permute_slices(hoNDArray< std::complex<float> >& data, arma::uvec indice)
{
    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > new_data;
    new_data.create(RO,E1, E2, CHA, N, S, SLC);

    size_t n, s;

    for (int i = 0; i < SLC; i++) {

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice[i]));
                std::complex<float> * out = &(new_data(0, 0, 0, 0, n, s, i));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

    data = new_data;

}





void GenericReconSMSPrepv0Gadget::remove_extra_dimension_and_permute_stack_dimension(hoNDArray< std::complex<float> >& data)
{
    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > FID_MB;
    FID_MB.create(RO, E1, E2, CHA,  N, S , lNumberOfStacks_ );

    //size_t nb_elements_multiband = data.get_number_of_elements()/MB_factor;

    size_t index_in;
    size_t index_out;

    size_t n, s;
    for (int a = 0; a < lNumberOfStacks_; a++)
    {
        index_in=indice_slice_mb[a];
        index_out=indice_mb[a];

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index_in));
                std::complex<float> * out = &(FID_MB(0, 0, 0, 0,  n, s,index_out));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

    data = FID_MB;

}




//sur les données single band
void GenericReconSMSPrepv0Gadget::create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);

    size_t MB=new_stack.get_size(4);
    size_t STK=new_stack.get_size(5);

    size_t n, s, a, m;
    size_t index;

    // copy of the data in the 8D array

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
                    std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));

                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }
}






void GenericReconSMSPrepv0Gadget::apply_relative_phase_shift(hoNDArray< std::complex<float> >& data)
{

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

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    center_k_space_E1=round(E1/2);

    GDEBUG_STREAM("  center_k_space_xml  "<<   center_k_space_xml  << " center_k_space_E1    "<<  center_k_space_E1  );

    // à définir dans SMS Base car c'est aussi utilisé dans SMSPostGadget
    arma::fvec index_imag = arma::linspace<arma::fvec>( 1, E1, E1 )  - center_k_space_E1 ;

    arma::cx_fvec phase;
    arma::cx_fvec shift_to_apply;

    phase.set_size(E1);
    phase.zeros();
    phase.set_imag(index_imag);

    size_t m,a,n,s,cha,e2,e1,ro;
    float caipi_factor;

    for (m = 0; m < MB_factor; m++) {

        caipi_factor=2*arma::datum::pi/(Blipped_CAIPI)*(m);
        GDEBUG_STREAM("  Blipped_CAIPI  "<<   Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
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





void GenericReconSMSPrepv0Gadget::apply_absolute_phase_shift(hoNDArray< std::complex<float> >& data)
{

    // fid_stack_SB_corrected(:, :, :, c, :, :, a, m)=fid_stack.SB_shift(:, :, :, c, :, :, a, m) * exp(1i*pi*z_offset_geo(index)/z_gap(1));

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    size_t m, a, n, s;
    size_t index;

    std::complex<double> ii(0,1);

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            std::complex<double> lala=  exp(arma::datum::pi*ii*z_offset_geo(index)/z_gap(0));
            std::complex<float>  lili=  static_cast< std::complex<float> >(lala) ;

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> *in = &(data(0, 0, 0, 0, m, a, n, s));

                    for (size_t ii = 0; ii < RO * E1 * E2 * CHA; ii++) {

                        data[ii] = data[ii]*lili  ;
                    }
                }
            }
        }
    }

}




GADGET_FACTORY_DECLARE(GenericReconSMSPrepv0Gadget)
}



