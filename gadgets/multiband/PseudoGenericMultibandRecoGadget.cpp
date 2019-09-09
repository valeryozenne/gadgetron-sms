/*
* PseudoGenericMultibandRecoGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "PseudoGenericMultibandRecoGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"
#include "mri_core_multiband.h"

namespace Gadgetron{

PseudoGenericMultibandRecoGadget::PseudoGenericMultibandRecoGadget() {
}

PseudoGenericMultibandRecoGadget::~PseudoGenericMultibandRecoGadget() {
}

int PseudoGenericMultibandRecoGadget::process_config(ACE_Message_Block *mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    ISMRMRD::StudyInformation study_info;

    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;  /* Number of slices in one measurement */ //MANDATORY
    lNumberOfAverage_= e_limits.average? e_limits.average->maximum+1 : 1;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;

    std::cout <<  h.encoding[0].encodingLimits.kspace_encoding_step_1->maximum << std::endl;
    std::cout <<  h.encoding[0].encodingLimits.kspace_encoding_step_1->minimum << std::endl;

    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);

    GDEBUG("Matrix size after reconstruction: %d, %d, %d\n", dimensions_[0], dimensions_[1], dimensions_[2]);

    ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;

    acceFactorE1_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_1);
    acceFactorE2_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_2);

    readout=dimensions_[0];
    encoding=dimensions_[1];

    //TODO probleme ne marche pas avec grappa 3
    last_scan_in_acs=(h.encoding[0].encodingLimits.kspace_encoding_step_1->center)+11;

    GDEBUG(" encoding center: %d \n", h.encoding[0].encodingLimits.kspace_encoding_step_1->center);
    GDEBUG(" last_scan_in_acs: %d \n", last_scan_in_acs);
    GDEBUG(" lNumberOfChannels_: %d \n", lNumberOfChannels_);

    deal_with_inline_or_offline_situation(h);

    str_home=GetHomeDirectory();

    debug=false;

    Gadgetron::DeleteTemporaryFiles( str_home, "Tempo/", "reco_sms*" );

    ///------------------------------------------------------------------------
    /// FR
    /// UK
    find_encoding_dimension_for_SMS_calculation();

    lNumberOfStacks_= lNumberOfSlices_/MB_factor_;

    slice_calibration.set_size(readout*lNumberOfChannels_, encoding, lNumberOfSlices_);
    folded_image.set_size(readout*lNumberOfChannels_, encoding, lNumberOfStacks_);

    slice_calibration_reduce.set_size(readout*lNumberOfChannels_, encoding, MB_factor_);
    folded_image_reduce.set_size(readout*lNumberOfChannels_, encoding, 1);

    slice_calibration_reduce_unblipped.set_size(readout*lNumberOfChannels_, encoding, MB_factor_);

    unfolded_image_with_blipped.set_size(readout*lNumberOfChannels_,encoding,MB_factor_);
    unfolded_image_with_blipped.zeros();

    unfolded_image_for_output.set_size(readout*lNumberOfChannels_, encoding, MB_factor_);
    unfolded_image_for_output.zeros();

    unfolded_image_for_output_all.set_size(readout*lNumberOfChannels_, encoding, lNumberOfSlices_);
    unfolded_image_for_output_all.zeros();

    set_kernel_parameters();

    // flag a définir dans le xml au minimum
    bool no_reordering=0;

    order_of_acquisition_mb = Gadgetron::map_interleaved_acquisitions_mb(lNumberOfStacks_,  no_reordering );

    order_of_acquisition_sb = Gadgetron::map_interleaved_acquisitions(lNumberOfSlices_,  no_reordering );

    indice_mb = sort_index( order_of_acquisition_mb );

    indice_sb = sort_index( order_of_acquisition_sb );

    slice_number_of_acquisition_mb.set_size(lNumberOfStacks_);
    slice_number_of_acquisition_mb.zeros();
    index_of_acquisition_mb.set_size(lNumberOfStacks_);
    index_of_acquisition_mb.zeros();

    for (unsigned long a = 0; a < lNumberOfStacks_; a++)
    {
        slice_number_of_acquisition_mb(a)=indice_sb(a);
    }

    //this is the number of the slices that will be acquired in mb
    std::cout <<  " this is the number of the slices that will be acquired in mb "<< std::endl;
    std::cout <<  slice_number_of_acquisition_mb << std::endl;

    for (unsigned long a = 0; a < lNumberOfStacks_; a++)
    {
        std::cout << " a "<< a <<   " order_of_acquisition_mb(a) "<< order_of_acquisition_mb(a) <<   "  acquisition order " <<  slice_number_of_acquisition_mb(order_of_acquisition_mb(a)) << std::endl;
        index_of_acquisition_mb(a)=slice_number_of_acquisition_mb(order_of_acquisition_mb(a));

        std::cout << " a "<< a << "  order_of_acquisition_sb(a)  "<< order_of_acquisition_sb(a) << std::endl;
    }

    // si c'est pair c'est vrai

    //this is the order in time of mb acquisition  << std::endl;
    std::cout <<" this is the order in time of the mb slices acquisition " << std::endl;
    std::cout <<  index_of_acquisition_mb << std::endl;


    gt_timer_.set_timing_in_destruction(false);
    gt_timer_local_.set_timing_in_destruction(false);

    // Reordering
    // Assuming interleaved SMS acquisition, define the acquired slices for each multislice band
    MapSliceSMS=Gadgetron::get_map_slice_single_band( MB_factor_,  lNumberOfStacks_,  order_of_acquisition_mb,  no_reordering);

    vec_MapSliceSMS=vectorise(MapSliceSMS);

    recon_crop.set_size(MB_factor_,1);
    recon_prep.set_size(MB_factor_,1);
    recon_reshape.set_size(MB_factor_,1);

    recon_crop_template.set_size(blocks_per_row, lNumberOfChannels_, blocks_per_column);
    recon_prep_template.set_size(blocks_per_row+4, lNumberOfChannels_, blocks_per_column+4);
    recon_reshape_template.set_size(readout*lNumberOfChannels_,e1_size_for_sms_calculation,1);

    recon_crop_template.zeros();
    recon_prep_template.zeros();
    recon_reshape_template.zeros();

    matrix_encoding.set_size(encoding,lNumberOfSlices_);
    matrix_encoding.zeros();

    acs_encoding.set_size(encoding,lNumberOfSlices_);
    acs_encoding.zeros();

    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        recon_crop(m,0) = recon_crop_template;
        recon_prep(m,0) = recon_prep_template;
        recon_reshape(m,0) = recon_reshape_template;
    }

    return GADGET_OK;
}




int PseudoGenericMultibandRecoGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    bool is_acq_single_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
    bool is_acq_multi_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;

    arma::cx_fvec tempo= as_arma_col(*m2->getObjectPtr());

    if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
    {
        std::cout << "  ISMRMRD_ACQ_IS_PHASECORR_DATA " << std::endl;
    }

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
        /// elles ont peut-être été blippés auparavant si nécessaire
        /// UK

        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }

    }
    else if (is_acq_single_band)
    {
        ///------------------------------------------------------------------------
        /// FR stockage du kspace  [ readout*coil , encoding , nbSlicesTotal  ]
        /// UK

        //slice_calibration.slice(slice).col(e1)=tempo;

        slice_calibration.slice(get_spatial_slice_number(slice)).col(e1)=tempo;

        matrix_encoding(e1,slice)++;

        if (is_last_scan_in_slice && slice==indice_sb(lNumberOfSlices_-1))
        {
            //GDEBUG("Sorting slice  \n");

            GDEBUG("Sorting slice %d  %d  %d \n", is_last_scan_in_slice , e1,  slice);

            /*for (unsigned long s = 0; s < lNumberOfSlices_; s++)
            {
                for (unsigned long e1 = 0; e1 < encoding; e1++)
                {
                //ceci est le numero de dynamique ie de 0 à N-1
                std::ostringstream indice_e1;
                indice_e1 << e1;
                str_e = indice_e1.str();

                std::ostringstream indice_slice;
                indice_slice << s;
                str_s = indice_slice.str();

                //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration.slice(s).col(e1)), str_home, "Tempo/", "slice_calibration_",  str_e, str_s,  ".bin");
                //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration_test.slice(s).col(e1)), str_home, "Tempo/", "slice_calibration_test_",  str_e, str_s,  ".bin");

                }
            }*/

            ///------------------------------------------------------------------------
            /// FR normalement boucle sur le nombre de groupe, hypthese que les slices sont conjointes
            /// UK

            for (unsigned int a = 0; a < lNumberOfStacks_; a++)
            {
                GDEBUG("Kernel calculation is starting \n");

                //std::cout <<   MapSliceSMS .col(a).t() << std::endl;

                // cette étape pourrait être évité en reordonnant bien avant les coupes de cette maniere
                // [1 5 9 3 7 11 0 4 8 2 6 12] et en extrayant des paquets de 3 successivement

                // mais pour l'instant on a reorganisé les coupes dans l'ordre
                // [1 2 3 4 5 6 7 8 9 10 11 12]
                // et on extrait les différents paquets dans reduce
                // 1er paquet [1 5 9]
                // 2nd paquet [3 7 11]

                for (unsigned int m = 0; m < MB_factor_; m++)
                {
                    slice_calibration_reduce.slice(m)=slice_calibration.slice(MapSliceSMS(m,a));
                }

                //int debut=position_debut_fin(0,grp);
                //int fin=position_debut_fin(1,grp);

                //GDEBUG("Groupe: %d , debut: %d   , fin: %d  \n", a ,  debut , fin);

                hoNDArray< std::complex<float> > copy;
                copy.create(readout*lNumberOfChannels_, encoding, MB_factor_);
                memcpy(&copy(0,0,0), &slice_calibration_reduce(0,0,0) ,sizeof(std::complex<float>)*readout*lNumberOfChannels_*encoding*MB_factor_);

                std::stringstream stk;
                stk << "_stack_" << a;
                gt_exporter_.export_array_complex(copy, "/tmp/gadgetron/SliceGrappa/slice_calibration_reduce0" + stk.str());

                calculate_kernel(slice_calibration_reduce);

                // on  calcule la kernel pour chaque slice du paquet et on retrit les données de la même manière
                // 1er paquet [1 5 9]
                // 2nd paquet [3 7 11]

                //ainsi kernel_all_slices est organisé dans l'ordre géométrique

                for (unsigned int m = 0; m < MB_factor_; m++)
                {
                    std::cout << "kernel a : " <<   " m : " << m  << "    MapSliceSMS(m,a)" <<  MapSliceSMS(m,a) << std::endl;

                    kernel_all_slices.slice(MapSliceSMS(m,a))=kernel.slice(m);
                }

                GDEBUG("Kernel calculation is ending \n");
            }
        }

        // setPhase(m1->getObjectPtr()->idx,1);

        m1->getObjectPtr()->idx.phase=1;

        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }
    }
    else if (is_acq_multi_band)
    {

        //if (e1==0)
        //{
        //std::cout << " slice "<< slice << "  indice_sb(slice) " << indice_mb(slice) << "  order_of_acquisition_sb(slice) " << order_of_acquisition_mb(slice) << "   " << indice_mb(lNumberOfSlices_-1) << endl;
        //std::cout << " slice "<< slice << "  indice_sb(slice) " << indice_sb(slice) << "  order_of_acquisition_sb(slice) " << order_of_acquisition_sb(slice) << "   " << indice_sb(lNumberOfSlices_-1) << endl;

        //std::cout << "index_mb " << indice_mb.t() << std::endl;
        //}

        if (is_first_scan_in_slice)
        {
            GDEBUG("  Buffering slice :  %d , spatial slice position : %d ,   stack number : %d ,   indice back : %d \n",   slice, get_spatial_slice_number(slice) , slice, indice_sb(slice));
        }

        folded_image.slice(get_spatial_slice_number(slice)).col(e1)=tempo;
        //folded_image_reduce.slice(order_of_acquisition_sb(slice)).col(e1)=tempo;

        //if (is_last_scan_in_slice && get_spatial_slice_number(slice)==(lNumberOfStacks_-1))
        if (is_last_scan_in_slice &&  (slice==index_of_acquisition_mb(lNumberOfStacks_-1)))
        {
            GDEBUG(" All folded slices have been read \n");
            //std::cout << "  get_spatial_slice_number(slice) " << get_spatial_slice_number(slice) << "  "<< (lNumberOfStacks_-1) << std::endl;
            std::cout << "  (slice) " << (slice) << "  "<< index_of_acquisition_mb(lNumberOfStacks_-1) << std::endl;

            /*for (unsigned long a = 0; a < lNumberOfStacks_; a++)
            {
                for (unsigned long e1 = 0; e1 < encoding; e1++)
                {
                //ceci est le numero de dynamique ie de 0 à N-1
                std::ostringstream indice_e1;
                indice_e1 << e1;
                str_e = indice_e1.str();

                std::ostringstream indice_stack;
                indice_stack << a;
                str_a = indice_stack.str();

                Gadgetron::SaveVectorOntheDisk(vectorise(folded_image.slice(a).col(e1)), str_home, "Tempo/", "folded_image_",  str_e, str_a,  ".bin");

                }
            }*/

            ///------------------------------------------------------------------------
            /// FR
            /// UK

            std::cout << "order_of_acquisition_sb " << order_of_acquisition_mb.t() << std::endl;

            for (unsigned int a = 0; a < lNumberOfStacks_; a++)
            {
                GDEBUG("SMS Defolding is starting \n");

                //std::cout << " a  "  <<  a << " order_of_acquisition_mb(a)  "  << order_of_acquisition_mb(a)  << std::endl;

                //std::cout << size(folded_image.slices(order_of_acquisition_mb(a), order_of_acquisition_mb(a)) ) << std::endl;

                calculate_unfolded_image(folded_image.slices(order_of_acquisition_mb(a), order_of_acquisition_mb(a)), MapSliceSMS.col(a));

                for (unsigned long m = 0; m < MB_factor_; m++)
                {
                    GDEBUG("Copy SMS: %d  %d, position_slice: %d   \n", a , m, MapSliceSMS(m,a));
                    unfolded_image_for_output_all.slice(MapSliceSMS(m,a))=unfolded_image_for_output.slice(m);
                }

                GDEBUG("SMS Defolding is ending \n");

                transfer_kspace_to_the_next_gadget(m1,m2, a);
            }
        }
    }

    return GADGET_OK;
}


int PseudoGenericMultibandRecoGadget::transfer_kspace_to_the_next_gadget(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2 , int a)
{
    for (unsigned long m = 0; m < MB_factor_; m++)
    {

        /*for (unsigned long i = 0; i < encoding; i++)
        {

            arma::cx_fvec output= unfolded_image_for_output_all.slice(MapSliceSMS(m,a)).col(i);

            //ceci est le numero de dynamique ie de 0 à N-1
            std::ostringstream indice_e1;
            indice_e1 << i;
            str_e = indice_e1.str();

            std::ostringstream indice_s;
            indice_s << indice_sb(MapSliceSMS(m,a));
            str_s = indice_s.str();

            Gadgetron::SaveVectorOntheDisk(output, str_home, "Tempo/", "reco_sms_",  str_e, str_s,  ".bin");
            Gadgetron::SaveVectorOntheDisk(output, str_home, "Tempo/", "reco_sms_",  str_e, str_s,  ".bin");
        }*/


        for (unsigned long i = 0; i < encoding; i++)
        {

            //cependant il faut changer le test MapSliceSMS(m,a) n'est pas correct
            if(matrix_encoding(i,MapSliceSMS(m,a))>0)  // ce flag sert a ne pas envoyer les lignes vide dans le cas du grappa
            {

                GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* cm1 =
                        new GadgetContainerMessage<ISMRMRD::AcquisitionHeader>();

                *cm1->getObjectPtr() = *m1->getObjectPtr();

                GadgetContainerMessage<hoNDArray<std::complex<float> >  > *cm2 =
                        new GadgetContainerMessage<hoNDArray<std::complex<float> >  >();

                boost::shared_ptr< std::vector<size_t> > dims = m2->getObjectPtr()->get_dimensions();

                try{cm2->getObjectPtr()->create(dims.get());}
                catch (std::runtime_error &err){
                    GEXCEPTION(err,"Unable to create unsigned short storage in Extract Magnitude Gadget");
                    return GADGET_FAIL;
                }

                arma::cx_fvec temp1= unfolded_image_for_output_all.slice(MapSliceSMS(m,a)).col(i);

                std::complex<float>* dst = cm2->getObjectPtr()->get_data_ptr();

                for (unsigned long j = 0; j < m2->getObjectPtr()->get_number_of_elements(); j++) {
                    dst[j]=temp1[j];
                }

                //if (i==61)
                //{
                //   std::cout<< " a:  "<< a <<"m: "<<  m <<  "  MapSliceSMS(m,a) "<< MapSliceSMS(m,a)<<  " indice_sb:  "<< indice_sb(MapSliceSMS(m,a))<< "  " << std::endl;
                //}

                // TODO ici il faut redonner le bon indice
                cm1->getObjectPtr()->idx.slice=indice_sb(MapSliceSMS(m,a));
                cm1->getObjectPtr()->idx.kspace_encode_step_1=i;

                if (i==flag_encoding_first_scan_in_slice) //TODO
                {
                    cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
                }
                else
                {
                    cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
                }

                if (i==flag_encoding_last_scan_in_slice)  //TODO
                {
                    cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
                }
                else
                {
                    cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
                }

                if((indice_sb(MapSliceSMS(m,a))==indice_sb(lNumberOfSlices_-1)) && (i==flag_encoding_last_scan_in_slice))
                {
                    std::cout << "ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION " << indice_sb(MapSliceSMS(m,a)) << " "<< indice_sb(lNumberOfSlices_-1) <<std::endl;
                    cm1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION);
                }
                else
                {
                    cm1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION);
                }

                cm1->cont(cm2);

                if( this->next()->putq(cm1) < 0 ){
                    GDEBUG("Failed to put message on queue\n");
                    return GADGET_FAIL;
                }
            }
        }
    }

    return 0;
}



void PseudoGenericMultibandRecoGadget::calculate_unfolded_image(arma::cx_fcube input, arma::ivec kernel_slice_position)
{

    ///------------------------------------------------------------------------
    /// FR il faut commencer par enlever les données grappa en trop [readout*coil encoding/in_plane_acce slices]
    /// UK
    arma::cx_fcube folded_image_ready=remove_data_if_inplane_grappa(input);

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube [readout coil encoding]
    /// UK
    arma::cx_fcube folded_image_reshape=preliminary_reshape(folded_image_ready.slice(0));

    ///------------------------------------------------------------------------
    /// FR on applique la fonction im2col
    /// UK
    arma::cx_fcube block_MB=im2col_version2(folded_image_reshape);

    block_MB.reshape( size(block_MB,0), size(block_MB,1)* size(block_MB,2),1 );

    block_MB.reshape( blocks_per_column, blocks_per_row, size(block_MB,1) );

    arma::cx_fcube lala(1,kernel_size*lNumberOfChannels_,1);
    arma::cx_frowvec lili(kernel_size*lNumberOfChannels_);


    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        GDEBUG(" kernel_slice_position for defolding will be : %d \n", kernel_slice_position(m));
    }

    for (unsigned long nidx1 = 0; nidx1 < blocks_per_column; nidx1++) //for nidx1=1:blocks_per_column
    {
        for (unsigned long nidx2 = 0; nidx2 < blocks_per_row; nidx2++) // for nidx2=1:blocks_per_row
        {
            for (unsigned int c = 0; c < lNumberOfChannels_; c++)   //for nidy2=1:nc2   number of channels
            {
                //recon(nidx1,nidx2,1,nidy2,nidy3,nidy4)=permute(block_MB(nidx1,nidx2,:),[1 3 2]) * kernel(:,nidy2,nidy3,nidy4) ;

                lala=reshape(block_MB.tube(nidx1,nidx2),1,kernel_size*lNumberOfChannels_,1);
                lili=lala.slice(0);

                for (unsigned long m = 0; m < MB_factor_; m++)
                {
                    recon_crop(m,0).slice(nidx1).col(c).row(nidx2)=lili*kernel_all_slices.slice(kernel_slice_position(m)).col(c);
                }
            }
        }
    }

    int index_e1;

    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        recon_prep(m,0).subcube(2,0,2,blocks_per_row+1,lNumberOfChannels_-1,blocks_per_column+1 )=recon_crop(m,0); //TODO verifier le padding
        recon_reshape(m,0)=reshape(recon_prep(m,0),readout*lNumberOfChannels_,e1_size_for_sms_calculation,1);

        for (unsigned long j = 0; j < e1_size_for_sms_calculation; j++) {

            if (acceFactorE1_>1)
            {
                index_e1=j*acceFactorE1_+1;
                //std::cout << " j " << j <<   "  index_e1 " << index_e1 << std::endl;
                unfolded_image_with_blipped.slice(m).col(index_e1)=recon_reshape(m,0).slice(0).col(j);
            }
            else
            {
                unfolded_image_with_blipped.slice(m).col(j)=recon_reshape(m,0).slice(0).col(j);
            }
        }
    }


    if (acceFactorE1_>1)
    {
        unfolded_image_for_output=unfolded_image_with_blipped;
    }
    else
    {
        unfolded_image_for_output=unfolded_image_with_blipped;
        //unfolded_image_for_output=shifting_of_multislice_data(unfolded_image_with_blipped, -Blipped_CAIPI_);
    }

    /*for (unsigned long c = 0; c < nbChannels; c++)
    {
        //ceci est le numero de dynamique ie de 0 à N-1
        std::ostringstream indice_coil;
        indice_coil << c;
        str_c = indice_coil.str();
        Gadgetron::SaveVectorOntheDisk(vectorise(recon_slice0_ok.slice(c)), str_home, "Tempo/", "reco_sms_",  str_c, "0",  ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(recon_slice1_ok.slice(c)), str_home, "Tempo/", "reco_sms_",  str_c, "1",  ".bin");
    }*/
}




void PseudoGenericMultibandRecoGadget::detect_flag(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1)
{
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_first_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_ENCODE_STEP1);
    bool is_last_in_encoding = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_ENCODE_STEP1);
    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;

    if (is_first_in_encoding)
    {
        flag_encoding_first_in_encoding=e1;
        //std::cout<< "  flag_encoding_last_in_encoding " << flag_encoding_last_in_encoding  << std::endl;
    }

    if (is_last_in_encoding)
    {
        flag_encoding_last_in_encoding=e1;
        //std::cout<< "  flag_encoding_last_in_encoding " << flag_encoding_last_in_encoding  << std::endl;
    }

    if (is_first_scan_in_slice)
    {
        flag_encoding_first_scan_in_slice=e1;
        //std::cout<< "  flag_encoding_first_scan_in_slice " << flag_encoding_first_scan_in_slice  << std::endl;
    }

    if (is_last_scan_in_slice)
    {
        flag_encoding_last_scan_in_slice=e1;
        //std::cout<< "  flag_encoding_last_scan_in_slice " << flag_encoding_last_scan_in_slice  << std::endl;
    }

}




void PseudoGenericMultibandRecoGadget::calculate_kernel(arma::cx_fcube input)
{

    ///------------------------------------------------------------------------
    /// FR nous sauvegardons le kspace sur le disque
    /// /// UK
    if (debug)
    {

        std::cout << "debug"<< std::endl;
        save_kspace_data(input,str_home, "Tempo/", "scs_");
    }

    ///------------------------------------------------------------------------
    /// FR il faut commencer par enlever les données grappa en trop [readout*coil encoding/in_plane_acce slice]
    /// UK
    arma::cx_fcube slice_calibration_nohole=remove_data_if_inplane_grappa(input);

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube de [readout*coil encoding slice] à  [encoding readout coil ] et on divise le resultat en field
    /// UK

    arma::field<arma::cx_fcube> block_SB(MB_factor_,1);

    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        block_SB(m,0) = im2col(preliminary_reshape(slice_calibration_nohole.slice(m)));
    }

    //arma::cx_fcube slice_calibration_reshape_slice_0=preliminary_reshape(slice_calibration_nohole.slice(0));
    //arma::cx_fcube slice_calibration_reshape_slice_1=preliminary_reshape(slice_calibration_nohole.slice(1));
    //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration_reshape_slice_0.slice(0)),str_home, "Tempo/", "kernel_slice_calibration_reshape_",  "0", "0", ".bin");
    //Gadgetron::SaveVectorOntheDisk(vectorise(slice_calibration_reshape_slice_1.slice(0)),str_home, "Tempo/", "kernel_slice_calibration_reshape_",  "0", "1", ".bin");
    //std::cout <<  "size(slice_calibration_reshape) " << std::endl;
    //std::cout <<  size(slice_calibration_reshape_slice_0) << std::endl;
    //std::cout <<  size(slice_calibration_reshape_slice_1) << std::endl;
    //arma::cx_fcube block_SB_reshape_slice_0=im2col(slice_calibration_reshape_slice_0);
    //arma::cx_fcube block_SB_reshape_slice_1=im2col(slice_calibration_reshape_slice_1);
    //std::cout <<  "size(block_SB_reshape_slice_0 et 1)" << std::endl;
    //std::cout <<  size(block_SB_reshape_slice_0) << std::endl;
    //std::cout <<  size(block_SB_reshape_slice_1) << std::endl;

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(block_SB(0,0).slice(0)),str_home, "Tempo/", "kernel_block_SB_reshape_",  "0", "0", ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(block_SB(1,0).slice(0)),str_home, "Tempo/", "kernel_block_SB_reshape_",  "0", "1", ".bin");
    }

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube de [] à  []
    /// UK

    int milieu=round(float(kernel_size)/2)-1;
    //GDEBUG("Le milieu du kernel est %d  \n",milieu );

    arma::cx_fcube missing_data_slice(nb_pixels_per_image, lNumberOfChannels_,MB_factor_);

    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        for (unsigned long p = 0; p < nb_pixels_per_image; p++)
        {
            for (unsigned long c = 0; c < lNumberOfChannels_; c++)
            {
                missing_data_slice(p,c,m)=block_SB(m,0)(milieu,p,c);
                //missing_data_slice(p,c,0)=block_SB_reshape_slice_0(milieu,p,c);
                //missing_data_slice(p,c,1)=block_SB_reshape_slice_1(milieu,p,c);
            }
        }
    }

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(missing_data_slice.slice(0)),str_home, "Tempo/", "kernel_missing_data_slice_",  "0",  ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(missing_data_slice.slice(1)),str_home, "Tempo/", "kernel_missing_data_slice_",  "1",  ".bin");
    }

    arma::cx_fcube measured_data=block_SB(0,0);
    measured_data.zeros();

    for (unsigned long m = 0; m < MB_factor_; m++)
    {
        measured_data=measured_data+block_SB(m,0);
    }

    if (debug)
    {
        for (unsigned long c = 0; c < lNumberOfChannels_; c++)
        {
            //ceci est le numero de dynamique ie de 0 à N-1
            std::ostringstream indice_coil;
            indice_coil << c;
            str_c = indice_coil.str();
            Gadgetron::SaveVectorOntheDisk(vectorise(measured_data.slice(c)), str_home, "Tempo/", "kernel_measured_data_",  str_c,  ".bin");
        }
    }

    ///------------------------------------------------------------------------
    /// FR on reformate dans un cube [pixel , kernel , coil]
    /// UK

    arma::cx_fcube measured_data_ready(nb_pixels_per_image, kernel_size, lNumberOfChannels_ );

    for (unsigned long k = 0; k < kernel_size; k++)
    {
        for (unsigned long j = 0; j < nb_pixels_per_image; j++)
        {
            for (unsigned long c = 0; c < lNumberOfChannels_; c++)
            {
                measured_data_ready(j,k,c)=measured_data(k,j,c);
            }
        }
    }

    ///------------------------------------------------------------------------
    /// FR son reformate dans un cube [pixel , kernel*coil , 1]
    /// UK

    measured_data_ready.reshape( size(measured_data_ready,0), size(measured_data_ready,1) * size(measured_data_ready,2) ,  1 );

    arma::cx_fmat measured_data_matrix=measured_data_ready.slice(0);

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(measured_data_matrix), str_home, "Tempo/", "kernel_measured_data_matrix_",  "0",  ".bin");
    }

    ///------------------------------------------------------------------------
    /// FR on prepare la résolution du systeme
    /// UK

    arma::cx_fmat A=(measured_data_matrix.t()*measured_data_matrix);

    ///------------------------------------------------------------------------
    /// FR résolution du systeme
    /// UK

    //Gadgetron::SaveVectorOntheDisk(vectorise(A),str_home, "Tempo/", "A_",  "0",  ".bin");

  //  gt_timer_.start("PseudoWIPGenericMultibandGadget::process  CMK ");
    arma::cx_fmat CMK = pinv(A)*measured_data_matrix.t();
  //  gt_timer_.stop();

    if (debug)
    {
        Gadgetron::SaveVectorOntheDisk(vectorise(CMK),str_home, "Tempo/", "kernel_CMK_",  "0",  ".bin");
    }

    arma::cx_fmat B=measured_data_matrix.t();

   // gt_timer_.start("PseudoWIPGenericMultibandGadget::process  CMK ");
    arma::cx_fmat X = solve( A, B, arma::solve_opts::fast );
   // gt_timer_.stop();

      //   Gadgetron::SaveVectorOntheDisk(vectorise(X),str_home, "Tempo/", "kernel_X_",  "0",  ".bin");



    ///------------------------------------------------------------------------
    /// FR écriture du kernel
    /// UK

    //TODO rajouter les différentes implémentations
    for (unsigned long nidx1 = 0; nidx1 < size(CMK,0); nidx1++)
    {
        for (unsigned long c = 0; c < lNumberOfChannels_; c++)
        {
            for (unsigned long m = 0; m < MB_factor_; m++)
            {
                kernel.slice(m).col(c).row(nidx1)= X.row(nidx1)*missing_data_slice.slice(m).col(c) ;
            }
        }
    }
}



arma::cx_fcube PseudoGenericMultibandRecoGadget::remove_data_if_inplane_grappa(arma::cx_fcube input)
{

    ///------------------------------------------------------------------------
    /// FR
    /// UK

    GDEBUG("Encoding size for sms calculation: %d\n", e1_size_for_sms_calculation);

    arma::cx_fcube  output(size(input,0),e1_size_for_sms_calculation,size(input,2) );

    if(acceFactorE1_>1)
    {
        GDEBUG("Changing encoding size for sms calculation from %d to %d\n",encoding, e1_size_for_sms_calculation);

        int indice;

        for (unsigned int s = 0; s < size(input,2); s++)
        {
            // std::cout << s << std::endl;
            // ici on peut ajouter une tableau qui dit si on continue ou pas

            for (unsigned int j = 1; j < size(input,1); j+=acceFactorE1_)
            {
                indice = ((j+1)/acceFactorE1_)-1;
                output.slice(s).col(indice)=input.slice(s).col(j);
            }
        }
    }
    else
    {
        GDEBUG("Keeping encoding size for sms calculation to %d\n", e1_size_for_sms_calculation);
        output=input;
    }

    return output;

}



arma::cx_fcube PseudoGenericMultibandRecoGadget::im2col(arma::cx_fcube input)
{
    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK

    int xB=blocks_per_column;
    int yB=blocks_per_row;

    arma::cx_fcube block_MB(kernel_size,xB*yB,lNumberOfChannels_ );
    block_MB.zeros();

    int rowIdx, colIdx;

    for (unsigned int c = 0; c < lNumberOfChannels_; c++)
    {
        for (unsigned int j = 0; j < yB; j++)
        {
            for (unsigned int i = 0; i < xB; i++)
            {
                rowIdx = i + j*xB;

                for (unsigned int yy = 0; yy < kernel_height.value(); yy++)
                {
                    for (unsigned int xx = 0; xx < kernel_width.value(); xx++)
                    {
                        colIdx = xx + yy*kernel_width.value();

                        //if (c==0 && i==0 && j==0)     std::cout <<   xx <<  " " <<  yy<< " "<<  colIdx <<std::endl;

                        block_MB(colIdx, rowIdx,c)=input(i+xx, j+yy,c);
                    }
                }
            }
        }
    }

    return block_MB;
}



arma::cx_fcube PseudoGenericMultibandRecoGadget::shifting_of_multislice_data(arma::cx_fcube input, int PE_shift)
{

    int center_k_space_sample=round(float(encoding)/2);

    arma::fvec index_imag = arma::linspace<arma::fvec>( 1, encoding, encoding )  - center_k_space_sample ;

    arma::cx_fvec phase(encoding);
    phase.zeros();
    phase.set_imag(index_imag);

    arma::cx_fcube output=input;
    output.zeros();

    double caipi_factor;

    for (unsigned int m = 0; m < MB_factor_; m++)
    {
        caipi_factor=2*arma::datum::pi/PE_shift*m; // if m==0 rien, if m==1 2*pi/shift

        GDEBUG("Blipped caipi shift: slice: %d , caipi_factor: %f \n", m, caipi_factor);

        arma::cx_fvec lala=exp(phase*caipi_factor);

        for (unsigned int r = 0; r < size(input,0); r++)
        {
            output.slice(m).row(r)=input.slice(m).row(r)%lala.t();
        }
    }

    return output;
}


arma::cx_fcube PseudoGenericMultibandRecoGadget::im2col_version2(arma::cx_fcube input)
{
    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK

    GDEBUG("Blocks_per_column:  %d,  blocks_per_row: %d, nb_pixels_per_image: %d\n", blocks_per_column, blocks_per_row,nb_pixels_per_image  );

    int xB=blocks_per_column;
    int yB=blocks_per_row;

    arma::cx_fcube block_MB(xB*yB,kernel_size,lNumberOfChannels_ );
    block_MB.zeros();

    int rowIdx, colIdx;

    for (unsigned int c = 0; c < lNumberOfChannels_; c++)
    {

        for (unsigned int j = 0; j < yB; j++)
        {
            for (unsigned int i = 0; i < xB; i++)
            {
                rowIdx = i + j*xB;

                for (unsigned int yy = 0; yy < kernel_height.value(); yy++)
                {
                    for (unsigned int xx = 0; xx < kernel_width.value(); xx++)
                    {
                        colIdx = xx + yy*kernel_width.value();

                        //if (c==0 && i==0 && j==0)     std::cout <<   xx <<  " " <<  yy<< " "<<  colIdx <<std::endl;

                        block_MB(rowIdx, colIdx,c)=input(i+xx, j+yy,c);
                    }
                }
            }
        }
    }

    return block_MB;
}



arma::cx_fcube PseudoGenericMultibandRecoGadget::preliminary_reshape(arma::cx_fmat input)
{
    ///------------------------------------------------------------------------
    /// FR
    /// UK

    arma::cx_fcube tempo(size(input,0),size(input,1),1 ) ;

    tempo.zeros();

    tempo.slice(0)=input;

    tempo.reshape( readout, lNumberOfChannels_, size(input,1) );

    arma::cx_fcube output(e1_size_for_sms_calculation , readout, lNumberOfChannels_ );

    for (unsigned int r = 0; r < readout; r++)
    {
        for (unsigned int e1 = 0; e1 < e1_size_for_sms_calculation; e1++)
        {
            for (unsigned int c = 0; c < lNumberOfChannels_; c++)
            {
                output(e1,r,c)= tempo(r,c,e1);
            }
        }
    }

    return output;

}


void PseudoGenericMultibandRecoGadget::find_encoding_dimension_for_SMS_calculation(void)
{
    if(acceFactorE1_>1)
    {
        e1_size_for_sms_calculation=round(encoding/acceFactorE1_);
    }
    else
    {
        e1_size_for_sms_calculation=encoding;
    }

    GDEBUG("Encoding size for sms calculation: %d\n", e1_size_for_sms_calculation);
}


void PseudoGenericMultibandRecoGadget::set_kernel_parameters(void)
{

    kernel_size=kernel_height.value()*kernel_width.value();

    kernel.set_size(kernel_size*lNumberOfChannels_,lNumberOfChannels_,MB_factor_);
    kernel_all_slices.set_size(kernel_size*lNumberOfChannels_,lNumberOfChannels_,lNumberOfSlices_);

    blocks_per_row=readout-kernel_width.value()+1;

    blocks_per_column=e1_size_for_sms_calculation-(kernel_height.value())+1;

    nb_pixels_per_image=blocks_per_column*blocks_per_row;

    GDEBUG("Blocks_per_column:  %d,  blocks_per_row: %d, nb_pixels_per_image: %d\n", blocks_per_column, blocks_per_row,nb_pixels_per_image  );

}


void PseudoGenericMultibandRecoGadget::get_multiband_parameters(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec store_info_special_card=Gadgetron::get_information_from_multiband_special_card(h);

    MB_factor_=store_info_special_card(1);
    MB_Slice_Inc_=store_info_special_card(3);
    Blipped_CAIPI_=store_info_special_card(4);

}


int PseudoGenericMultibandRecoGadget::get_spatial_slice_number(int slice)
{
    int value=order_of_acquisition_sb(slice);

    return value;
}



void PseudoGenericMultibandRecoGadget::deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h)
{
    if (doOffline.value()==1)
    {
        MB_factor_=MB_factor.value();
        Blipped_CAIPI_=Blipped_CAIPI.value();
        MB_Slice_Inc_=MB_Slice_Inc.value();
    }
    else
    {
        get_multiband_parameters(h);
    }
}



GADGET_FACTORY_DECLARE(PseudoGenericMultibandRecoGadget)
}
